#!/usr/bin/env python3
import os
import cv2
import glob
import argparse
import motmetrics as mm
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd

from groundingdino.util.inference import load_model, predict
from tracker.byte_tracker import BYTETracker as ByteTrackBaseline
from tracker.tracker_w_clip import BYTETracker as ByteTrackCLIP  # your CLIP-fused tracker
import clip
from torch.cuda.amp import autocast

# =========================
# Optuna-picked hyperparams (truncated as requested)
# =========================
BOX_THRESHOLD   = 0.42
TEXT_THRESHOLD  = 0.50
TRACK_THRESH    = 0.41
MATCH_THRESH    = 0.87
TRACK_BUFFER    = 198

# Other static config
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "car. pedestrian."
MIN_BOX_AREA = 10
FRAME_RATE = 10

# CLIP-fused specific knobs (kept stable across runs; not part of Optuna set)
CLIP_LAMBDA_WEIGHT = 0.25
CLIP_TEXT_SIM_THRESH = 0.00  # keep disabled for stability; turn on later if desired (e.g., 0.20)

# =========================
# Preprocessing
# =========================
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def letterbox(tensor, size):
    _, h, w = tensor.shape
    scale = min(size[0] / h, size[1] / w)
    resized = F.interpolate(tensor.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False)[0]
    pad_h = size[0] - resized.shape[1]
    pad_w = size[1] - resized.shape[2]
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return F.pad(resized, (left, right, top, bottom), value=0.0)

def preprocess_frame(frame, use_fp16=False):
    h, w = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).cuda()
    tensor = letterbox(tensor, size=(h, w))
    return tensor.half() if use_fp16 else tensor

# =========================
# Box conversion + filtering
# =========================
def _is_xyxy_pixels(arr):
    if isinstance(arr, np.ndarray):
        return arr.size > 0 and float(np.max(arr)) > 1.5
    return arr.numel() > 0 and float(arr.max().item()) > 1.5

def nms_xyxy(dets, iou_thresh=0.6):
    if dets.size == 0:
        return []
    x1, y1, x2, y2, s = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def convert_boxes_robust(boxes, logits, W, H,
                         min_side=8, max_area_frac=0.35, max_aspect_ratio=4.0, nms_iou=0.6):
    """Accept normalized cxcywh or pixel xyxy; return np.float32 [N,5] with filters + NMS."""
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().float().numpy()
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().float().numpy()

    dets = []
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    def push(x1, y1, x2, y2, score):
        x1 = max(0.0, min(float(x1), W - 1.0))
        y1 = max(0.0, min(float(y1), H - 1.0))
        x2 = max(0.0, min(float(x2), W - 1.0))
        y2 = max(0.0, min(float(y2), H - 1.0))
        if x2 <= x1 or y2 <= y1:
            return
        w, h = x2 - x1, y2 - y1
        if w < min_side or h < min_side:
            return
        area = w * h
        if area > max_area_frac * (W * H):
            return
        ar = max(w / (h + 1e-6), h / (w + 1e-6))
        if ar > max_aspect_ratio:
            return
        dets.append([x1, y1, x2, y2, float(score)])

    if _is_xyxy_pixels(boxes):
        for (x1, y1, x2, y2), score in zip(boxes, logits):
            push(x1, y1, x2, y2, score)
    else:
        for (cx, cy, w, h), score in zip(boxes, logits):
            cx, cy, w, h = map(float, (cx, cy, w, h))
            if w <= 0 or h <= 0: 
                continue
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            push(x1, y1, x2, y2, score)

    if len(dets) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    dets = np.asarray(dets, dtype=np.float32)
    keep = nms_xyxy(dets, iou_thresh=nms_iou)
    return dets[keep] if len(keep) else np.zeros((0, 5), dtype=np.float32)

# =========================
# CLIP helpers
# =========================
def parse_classes(prompt: str):
    return [c.strip() for c in prompt.split('.') if c.strip()]

def build_text_embeddings(clip_model, device, classes):
    with torch.no_grad():
        tokens = clip.tokenize(classes).to(device)
        with torch.cuda.amp.autocast(enabled=False):
            te = clip_model.encode_text(tokens)
        te = F.normalize(te.float(), dim=-1).contiguous()
    return te  # [C,D] fp32 on device

def build_image_embedding(clip_model, clip_preprocess, device, crop_bgr):
    crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    clip_input = clip_preprocess(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            emb = clip_model.encode_image(clip_input)  # [1,D] fp32
        emb = F.normalize(emb, dim=-1).squeeze(0).float().cpu()
    return emb  # CPU fp32 [D]

# =========================
# GT conversion (KITTI → MOT)
# =========================
def combine_gt_local(label_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for gt_file in glob.glob(os.path.join(label_folder, "*.txt")):
        seq = os.path.splitext(os.path.basename(gt_file))[0]
        out_path = os.path.join(out_folder, f"{seq}.txt")
        with open(gt_file) as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                parts = line.split()
                frame = int(parts[0]); tid=int(parts[1]); cls=parts[2]
                if cls not in ("Car","Pedestrian"): 
                    continue
                x1,y1,x2,y2 = map(float, parts[6:10])
                w,h = x2-x1, y2-y1
                f_out.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

# =========================
# Inference & tracking (both trackers)
# =========================
def run_compare_local(img_folder, out_root, use_fp16=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device).eval()

    # CLIP (for CLIP-fused tracker)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    classes = parse_classes(TEXT_PROMPT) or ["object"]
    text_emb = build_text_embeddings(clip_model, device, classes)  # [C,D] fp32

    # Output subfolders
    out_base = os.path.join(out_root, "baseline")
    out_clip = os.path.join(out_root, "clip_fused")
    out_vis  = os.path.join(out_root, "vis")
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(out_clip, exist_ok=True)
    os.makedirs(out_vis, exist_ok=True)

    # Process each sequence
    for seq in sorted(os.listdir(img_folder)):
        seq_path = os.path.join(img_folder, seq)
        if not os.path.isdir(seq_path):
            continue

        # Initialize trackers (same hyperparams for both, as requested)
        targs_base = argparse.Namespace(
            track_thresh=TRACK_THRESH,
            track_buffer=TRACK_BUFFER,
            match_thresh=MATCH_THRESH,
            aspect_ratio_thresh=10.0,
            min_box_area=MIN_BOX_AREA,
            mot20=False
        )
        targs_clip = argparse.Namespace(
            track_thresh=TRACK_THRESH,
            track_buffer=TRACK_BUFFER,
            match_thresh=MATCH_THRESH,
            lambda_weight=CLIP_LAMBDA_WEIGHT,
            text_sim_thresh=CLIP_TEXT_SIM_THRESH,
            aspect_ratio_thresh=10.0,
            min_box_area=MIN_BOX_AREA,
            mot20=False
        )
        tracker_base = ByteTrackBaseline(targs_base, frame_rate=FRAME_RATE)
        tracker_clip = ByteTrackCLIP(targs_clip, frame_rate=FRAME_RATE)

        out_file_base = os.path.join(out_base, f"{seq}.txt")
        out_file_clip = os.path.join(out_clip, f"{seq}.txt")
        os.makedirs(os.path.dirname(out_file_base), exist_ok=True)
        os.makedirs(os.path.dirname(out_file_clip), exist_ok=True)

        # For visualization pass, we’ll also collect per-frame drawn frames
        vis_path_base = os.path.join(out_vis, f"{seq}_baseline.mp4")
        vis_path_clip = os.path.join(out_vis, f"{seq}_clipfused.mp4")
        writer_base = None
        writer_clip = None

        with open(out_file_base, "w") as f_base, open(out_file_clip, "w") as f_clip:
            frame_names = sorted(os.listdir(seq_path))
            H = W = None
            for frame_name in frame_names:
                frame_id = int(os.path.splitext(frame_name)[0])
                img = cv2.imread(os.path.join(seq_path, frame_name))
                if img is None:
                    continue
                H, W = img.shape[:2] if H is None else (H, W)
                tensor = preprocess_frame(img, use_fp16).to(device)

                with torch.no_grad(), autocast(enabled=use_fp16):
                    boxes, logits, _ = predict(
                        model=model, image=tensor, caption=TEXT_PROMPT,
                        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD
                    )
                dets = convert_boxes_robust(boxes, logits, W, H)  # np.float32 [N,5]

                # ===== Baseline tracker =====
                if dets.size:
                    tracks_base = tracker_base.update(dets, [H, W], [H, W])
                else:
                    tracks_base = []
                # write MOT lines
                for t in tracks_base:
                    x,y,w,h = t.tlwh; tid = t.track_id
                    if w*h > MIN_BOX_AREA:
                        f_base.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

                # ===== CLIP-fused tracker =====
                if dets.size:
                    detection_embeddings = []
                    for d in dets:
                        x1, y1, x2, y2, _ = d.tolist()
                        xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
                        crop = img[yi1:yi2, xi1:xi2]
                        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                            detection_embeddings.append(None)
                            continue
                        emb = build_image_embedding(clip_model, clip_preprocess, device, crop)
                        detection_embeddings.append(emb)
                    tracks_clip = tracker_clip.update(
                        detections=dets,
                        detection_embeddings=detection_embeddings,
                        img_info=(H, W),
                        text_embedding=text_emb,
                        class_names=classes
                    )
                else:
                    tracks_clip = []
                # write MOT lines
                for t in tracks_clip:
                    x,y,w,h = t.tlwh; tid = t.track_id
                    if w*h > MIN_BOX_AREA:
                        f_clip.write(f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

                # ===== Visualization (draw IDs) =====
                # lazily create writers
                if writer_base is None:
                    writer_base = cv2.VideoWriter(
                        vis_path_base, cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, (W, H)
                    )
                    writer_clip = cv2.VideoWriter(
                        vis_path_clip, cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, (W, H)
                    )
                frame_b = draw_tracks(img.copy(), tracks_base, label="Baseline")
                frame_c = draw_tracks(img.copy(), tracks_clip, label="CLIP-Fused")
                writer_base.write(frame_b)
                writer_clip.write(frame_c)

        if writer_base is not None:
            writer_base.release()
        if writer_clip is not None:
            writer_clip.release()

        print(f"[{seq}] saved: baseline -> {out_file_base}, clip_fused -> {out_file_clip}")
        print(f"[{seq}] videos: {vis_path_base}, {vis_path_clip}")

    return out_base, out_clip, out_vis

# =========================
# Drawing helpers
# =========================
def draw_tracks(image, tracks, label=""):
    if label:
        cv2.putText(image, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    for t in tracks:
        x, y, w, h = map(int, t.tlwh)
        tid = int(t.track_id)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"ID:{tid}", (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# =========================
# Evaluation & comparison
# =========================
def evaluate_folder(gt_folder, res_folder):
    mh = mm.metrics.create()
    metrics = [
        'num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr',
        'precision', 'recall', 'num_switches',
        'mostly_tracked', 'mostly_lost', 'num_fragmentations',
        'num_false_positives', 'num_misses', 'num_objects'
    ]
    summaries = []
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.txt")))
    res_files = sorted(glob.glob(os.path.join(res_folder, "*.txt")))
    gt_map = {os.path.basename(p): p for p in gt_files}
    res_map = {os.path.basename(p): p for p in res_files}

    common = sorted(set(gt_map.keys()) & set(res_map.keys()))
    for name in common:
        gt = mm.io.loadtxt(gt_map[name], fmt='mot15-2D', min_confidence=1)
        res = mm.io.loadtxt(res_map[name], fmt='mot15-2D')
        acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=0.5)
        summary = mh.compute(acc, metrics=metrics, name=name[:-4])
        summaries.append(summary)

    df = pd.concat(summaries) if summaries else pd.DataFrame()
    if not df.empty:
        avg = df.mean(numeric_only=True)
        avg.name = 'AVG'
        df = pd.concat([df, avg.to_frame().T], axis=0)
    return df

def side_by_side(df_base, df_clip):
    idx = sorted(set(df_base.index) | set(df_clip.index))
    base = df_base.reindex(idx)
    clip = df_clip.reindex(idx)
    cols = base.columns
    out = pd.DataFrame(index=idx)
    for c in cols:
        out[(c, 'Baseline')] = base[c]
        out[(c, 'CLIP-Fused')] = clip[c]
        try:
            out[(c, 'Δ')] = out[(c, 'CLIP-Fused')] - out[(c, 'Baseline')]
        except Exception:
            out[(c, 'Δ')] = np.nan
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out

# =========================
# Main
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default="/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/image_02")
    parser.add_argument('--labels', default="/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/label_02")
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    # Output directory
    if args.outdir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
        run_outdir = os.path.join("outputs_compare", timestamp)
    else:
        run_outdir = args.outdir
    os.makedirs(run_outdir, exist_ok=True)
    out_gt = os.path.join(run_outdir, 'gt_local')
    print(f"\nAll outputs for this run will be saved in: {run_outdir}\n")

    # Prepare GT as MOT
    combine_gt_local(args.labels, out_gt)

    # Run inference + tracking for both trackers (same hyperparams)
    out_base, out_clip, out_vis = run_compare_local(
        img_folder=args.images,
        out_root=run_outdir,
        use_fp16=args.fp16
    )

    # Evaluate both
    print("\n===== EVALUATING: BASELINE =====")
    df_base = evaluate_folder(out_gt, out_base)
    print(df_base)

    print("\n===== EVALUATING: CLIP-FUSED =====")
    df_clip = evaluate_folder(out_gt, out_clip)
    print(df_clip)

    # Side-by-side comparison
    print("\n===== COMPARISON (Baseline vs CLIP-Fused, with Δ = CLIP - Baseline) =====")
    comp = side_by_side(df_base, df_clip)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(comp)

    # Save tables (readable)
    csv_base = os.path.join(run_outdir, "baseline_metrics.csv")
    csv_clip = os.path.join(run_outdir, "clip_fused_metrics.csv")
    csv_comp = os.path.join(run_outdir, "comparison_metrics.csv")
    df_base.to_csv(csv_base)
    df_clip.to_csv(csv_clip)
    comp.to_csv(csv_comp)
    print(f"\nSaved metrics:\n- {csv_base}\n- {csv_clip}\n- {csv_comp}")
    print(f"Visualization videos in: {out_vis}")
