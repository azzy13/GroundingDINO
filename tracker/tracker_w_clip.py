import numpy as np
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from tracker import matching
from .basetrack import BaseTrack, TrackState

_EPS = 1e-6

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = float(score)
        self.tracklet_len = 0
        self.embedding = None  # CPU torch.FloatTensor [D], unit norm

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True   # activate immediately so boxes draw
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = float(new_track.score)
        if new_track.embedding is not None:
            self.embedding = new_track.embedding.detach().float().cpu()
            self.embedding = self.embedding / (self.embedding.norm() + _EPS)

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = float(new_track.score)

        if new_track.embedding is not None:
            new_e = new_track.embedding.detach().float().cpu()
            if self.embedding is None:
                self.embedding = new_e
            else:
                self.embedding = 0.95 * self.embedding + 0.05 * new_e
            self.embedding = self.embedding / (self.embedding.norm() + _EPS)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-12)
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'

class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0
        self.args = args
        self.det_thresh = float(args.track_thresh)  # no extra +0.10 so inits happen
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, detections, detection_embeddings, img_info, text_embedding, class_names=None):
        """
        detections: np.float32 [N,5] -> [x1,y1,x2,y2,score]
        detection_embeddings: list of None or CPU FloatTensor [D] (unit norm)
        text_embedding: torch.FloatTensor [C,D] (unit norm, on device, FP32)
        """
        self.frame_id += 1
        activated_starcks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        if detections is None or len(detections) == 0:
            for track in self.tracked_stracks:
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost_stracks.append(track)
            self._finalize_lists(activated_starcks, refind_stracks, lost_stracks, removed_stracks)
            return [t for t in self.tracked_stracks if t.is_activated]

        dets_np = detections.astype(np.float32, copy=False)
        scores = dets_np[:, 4]
        bboxes = dets_np[:, :4]

        remain_inds = scores > self.args.track_thresh
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        det_embeddings_keep = [detection_embeddings[i] for i in range(len(detection_embeddings)) if remain_inds[i]]

        # Text gate disabled by default (threshold = 0.0): keep all
        filtered_dets, filtered_scores, filtered_embeddings = [], [], []
        te = text_embedding.to(dtype=torch.float32, copy=False)
        for i, emb in enumerate(det_embeddings_keep):
            # If you later enable: compute max_sim and check against args.text_sim_thresh
            filtered_dets.append(dets[i])
            filtered_scores.append(scores_keep[i])
            filtered_embeddings.append(emb)

        detections_trk = [STrack(STrack.tlbr_to_tlwh(tlbr), s)
                          for (tlbr, s) in zip(filtered_dets, filtered_scores)]
        for i, d in enumerate(detections_trk):
            if filtered_embeddings[i] is not None:
                fe = filtered_embeddings[i].detach().float().cpu()
                d.embedding = fe / (fe.norm() + _EPS)

        unconfirmed, tracked_stracks = [], []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = matching.embedding_iou_distance(
            strack_pool, detections_trk, lambda_weight=self.args.lambda_weight, adaptive=True
        ).astype(np.float32)

        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections_trk)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_trk[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists2 = matching.iou_distance(r_tracked_stracks, detections_trk).astype(np.float32)
        matches2, u_track2, _ = matching.linear_assignment(dists2, thresh=0.5)

        for itracked, idet in matches2:
            track = r_tracked_stracks[itracked]
            det = detections_trk[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track2:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections_left = [detections_trk[i] for i in u_detection]
        dists3 = matching.iou_distance(unconfirmed, detections_left).astype(np.float32)
        if not self.args.mot20:
            dists3 = matching.fuse_score(dists3, detections_left)
        matches3, u_unconfirmed, u_detection2 = matching.linear_assignment(dists3, thresh=0.7)

        for itracked, idet in matches3:
            unconfirmed[itracked].update(detections_left[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection2:
            track = detections_left[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        return [track for track in self.tracked_stracks if track.is_activated]

    def _finalize_lists(self, activated, refind, lost, removed):
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)

def joint_stracks(tlista, tlistb):
    exists, res = {}, []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb).astype(np.float32)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
