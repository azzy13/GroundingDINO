#!/usr/bin/env python3
"""
Clean refactored worker for GroundingDINO + Tracker evaluation.

Features:
  - Supports bytetrack, clip, and smartclip trackers
  - ReferKITTI GT visualization with YOLO-style labels
  - Referring detection filter for referring expression tracking
  - Optional video output with GT boxes
  - Multi-GPU dispatch support
"""
from __future__ import annotations

import os
import sys
import argparse
import importlib
import subprocess
from typing import Dict, Tuple, Optional, Iterable, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.cuda.amp import autocast
import clip
import pandas as pd

from groundingdino.util.inference import load_model, predict
from demo.florence2_adapter import Florence2Detector

# ============================
# Configuration Defaults
# ============================
DEFAULT_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
DEFAULT_WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
DEFAULT_TEXT_PROMPT = "car. pedestrian."
DEFAULT_MIN_BOX_AREA = 10
DEFAULT_FRAME_RATE = 10

TRACKER_REGISTRY: Dict[str, Tuple[str, str]] = {
    "bytetrack": ("tracker.byte_tracker", "BYTETracker"),
    "clip": ("tracker.tracker_w_clip", "CLIPTracker"),
    "smartclip": ("tracker.tracker_smart_clip", "SmartCLIPTracker"),
}


# ============================
# Utility Functions
# ============================
def build_normalize_transform():
    """Build image normalization transform with max 800px short side."""
    def resize_if_needed(img):
        w, h = img.size
        short_side = min(w, h)
        if short_side > 800:
            scale = 800 / short_side
            return img.resize((int(w * scale), int(h * scale)))
        return img

    return T.Compose([
        T.Lambda(resize_if_needed),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def parse_frame_id(frame_name: str) -> int:
    """Extract integer frame ID from filename (e.g., '000001.jpg' -> 1)."""
    stem = os.path.splitext(frame_name)[0]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse frame id from: {frame_name}")
    return int(digits)


def convert_dino_to_xyxy(boxes: Iterable, logits: Iterable, W: int, H: int) -> np.ndarray:
    """Convert DINO boxes (cx,cy,w,h normalized) to [x1,y1,x2,y2,score]."""
    dets = []
    for box, logit in zip(boxes, logits):
        cx, cy, w, h = box
        if w <= 0 or h <= 0:
            continue
        score = float(logit)
        x1 = (cx - w / 2.0) * W
        y1 = (cy - h / 2.0) * H
        x2 = (cx + w / 2.0) * W
        y2 = (cy + h / 2.0) * H
        dets.append([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2), score])
    return np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)


def parse_kv_list(kv_list):
    """Parse --tracker_kv key=val arguments into typed dict."""
    out = {}
    for kv in kv_list or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            elif "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


# ============================
# ReferKITTI GT Helpers
# ============================
def load_referkitti_labels(label_path: str) -> List[Dict]:
    """
    Load YOLO-style labels from ReferKITTI.

    Format: class_id track_id x_left_norm y_top_norm width_norm height_norm

    NOTE: Despite field names, ReferKITTI stores TOP-LEFT coordinates, not center coordinates!

    Returns:
        List of dicts with keys: class_id, track_id, x_center, y_center, width, height
        (names kept as x_center/y_center for compatibility, but values are actually top-left)
    """
    if not os.path.isfile(label_path):
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                labels.append({
                    "class_id": int(parts[0]),
                    "track_id": int(float(parts[1])),
                    "x_center": float(parts[2]),
                    "y_center": float(parts[3]),
                    "width": float(parts[4]),
                    "height": float(parts[5]),
                })
            except ValueError:
                continue
    return labels


def draw_referkitti_gt_boxes(frame: np.ndarray, label_path: str, target_ids: Optional[set] = None) -> np.ndarray:
    """
    Draw ReferKITTI GT boxes on frame.

    Args:
        frame: BGR image
        label_path: Path to YOLO-style label file
        target_ids: Optional set of track IDs to highlight

    Returns:
        Frame with GT boxes drawn
    """
    labels = load_referkitti_labels(label_path)
    H, W = frame.shape[:2]

    for lab in labels:
        tid = lab["track_id"]
        if target_ids is not None and tid not in target_ids:
            continue

        # Convert normalized coords to pixel bbox
        # NOTE: Despite the dict key names, these are TOP-LEFT coords, not center!
        x_left = lab["x_center"] * W
        y_top = lab["y_center"] * H
        bw = lab["width"] * W
        bh = lab["height"] * H
        x1, y1 = int(x_left), int(y_top)
        x2, y2 = int(x1 + bw), int(y1 + bh)

        # Draw bbox and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"GT ID:{tid}", (int(x1 + bw), y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame


# ============================
# Referring Detection Filter
# ============================
class ReferringDetectionFilter:
    """
    CLIP-based filter for referring expression tracking.

    Filters GroundingDINO detections by CLIP text-image similarity
    to keep only the most relevant objects for a referring expression.
    """

    def __init__(
        self,
        clip_model,
        clip_preprocess,
        text_embedding: torch.Tensor,
        mode: str = "topk",
        topk: int = 3,
        threshold: float = 0.0,
        pad: int = 4,
        device: str = "cuda"
    ):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        # Ensure text_embedding is 1D [D] or 2D [1, D]
        text_embedding = text_embedding.to(device)
        if text_embedding.dim() == 3:
            text_embedding = text_embedding.squeeze(0)
        if text_embedding.dim() == 2 and text_embedding.size(0) == 1:
            text_embedding = text_embedding.squeeze(0)
        self.text_embedding = text_embedding
        self.mode = mode.lower()
        self.topk = int(topk)
        self.threshold = float(threshold)
        self.pad = int(pad)
        self.device = device
        self.total_dets_in = 0
        self.total_dets_out = 0

    def filter(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
        """Filter detections based on CLIP similarity."""
        if self.mode == "none" or dets_xyxy.size == 0:
            return dets_xyxy

        self.total_dets_in += len(dets_xyxy)
        similarities = self._compute_similarities(frame_bgr, dets_xyxy)

        if self.mode == "topk":
            k = min(self.topk, len(dets_xyxy))
            top_indices = np.argsort(similarities)[-k:][::-1]
            filtered = dets_xyxy[top_indices]
        elif self.mode == "threshold":
            filtered = dets_xyxy[similarities >= self.threshold]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.total_dets_out += len(filtered)
        return filtered

    def _compute_similarities(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
        """
        Compute CLIP text-image similarity for each detection using crop+full-image fusion.

        This combines crop and full image embeddings to provide spatial context for
        expressions like "car on the left" or "rightmost vehicle".
        """
        if dets_xyxy.size == 0:
            return np.array([])

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Compute full image embedding once for spatial context
        full_img_pil = Image.fromarray(rgb)
        full_img_tensor = self.clip_preprocess(full_img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            full_img_emb = F.normalize(self.clip_model.encode_image(full_img_tensor), dim=-1).float()

        crops, valid_indices = [], []
        for i, (x1, y1, x2, y2, _) in enumerate(dets_xyxy):
            xi1 = max(0, int(x1) - self.pad)
            yi1 = max(0, int(y1) - self.pad)
            xi2 = min(W, int(x2) + self.pad)
            yi2 = min(H, int(y2) + self.pad)

            if xi2 > xi1 and yi2 > yi1 and (xi2 - xi1) >= 10 and (yi2 - yi1) >= 10:
                crops.append(self.clip_preprocess(Image.fromarray(rgb[yi1:yi2, xi1:xi2])))
                valid_indices.append(i)

        if not crops:
            return np.zeros(len(dets_xyxy), dtype=np.float32)

        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            crop_embs = F.normalize(self.clip_model.encode_image(batch), dim=-1).float()

        # Combine crop embeddings with full image embedding for spatial context
        combined_embs = F.normalize((crop_embs + full_img_emb) / 2.0, dim=-1)

        sims = F.cosine_similarity(
            self.text_embedding.unsqueeze(0).expand(len(combined_embs), -1),
            combined_embs, dim=-1
        ).cpu().numpy()

        all_sims = np.zeros(len(dets_xyxy), dtype=np.float32)
        for i, valid_i in enumerate(valid_indices):
            all_sims[valid_i] = sims[i]
        return all_sims

    def get_stats(self) -> dict:
        """Return filtering statistics."""
        retention = self.total_dets_out / self.total_dets_in if self.total_dets_in > 0 else 0.0
        return {
            "total_in": self.total_dets_in,
            "total_out": self.total_dets_out,
            "retention_rate": retention,
            "mode": self.mode
        }


# ============================
# Worker Class
# ============================
class Worker:
    """
    GroundingDINO + Tracker evaluation worker.

    Supports multiple tracker types, ReferKITTI GT visualization,
    referring expression filtering, and video output.
    """

    def __init__(
        self,
        *,
        # Detector config
        config_path: str = DEFAULT_CONFIG_PATH,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        text_prompt: str = DEFAULT_TEXT_PROMPT,
        detector: str = "dino",
        box_thresh: float = 0.35,
        text_thresh: float = 0.25,
        use_fp16: bool = False,
        device: Optional[str] = None,
        # Tracker config
        tracker_type: str = "bytetrack",
        tracker_kwargs: Optional[dict] = None,
        # Referring filter
        referring_mode: str = "none",
        referring_topk: int = 3,
        referring_thresh: float = 0.0,
        # Misc
        frame_rate: int = DEFAULT_FRAME_RATE,
        min_box_area: int = DEFAULT_MIN_BOX_AREA,
        verbose_first_n_frames: int = 5,
        save_video: bool = False,
        show_gt_boxes: bool = False,
        dataset_type: str = "mot",  # "mot" or "referkitti"
        referkitti_data_root: Optional[str] = None,  # Path to ReferKITTI root (for GT labels)
        target_object_ids: Optional[List[int]] = None,  # For referring expressions - which object IDs to show as GT
    ):
        self.text_prompt = text_prompt
        self.box_thresh = float(box_thresh)
        self.text_thresh = float(text_thresh)
        self.use_fp16 = bool(use_fp16)
        self.frame_rate = int(frame_rate)
        self.min_box_area = int(min_box_area)
        self.save_video = bool(save_video)
        self.show_gt_boxes = bool(show_gt_boxes)
        self.dataset_type = dataset_type
        self.referkitti_data_root = referkitti_data_root
        self.target_object_ids = set(target_object_ids) if target_object_ids else None
        self.verbose_first_n_frames = int(verbose_first_n_frames)

        # Device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Detector
        self.detector_kind = detector
        if self.detector_kind == "dino":
            self.dino_model = load_model(config_path, weights_path)
            if hasattr(self.dino_model, "to"):
                self.dino_model = self.dino_model.to(self.device)
        else:
            self.florence = Florence2Detector(
                model_id="microsoft/Florence-2-large",
                device=self.device,
                fp16=self.use_fp16
            )

        self._transform = build_normalize_transform()

        # Tracker
        tracker_kwargs = dict(tracker_kwargs or {})
        tracker_args = argparse.Namespace(
            track_thresh=tracker_kwargs.pop("track_thresh", 0.5),
            track_buffer=tracker_kwargs.pop("track_buffer", 30),
            match_thresh=tracker_kwargs.pop("match_thresh", 0.8),
            aspect_ratio_thresh=tracker_kwargs.pop("aspect_ratio_thresh", 10.0),
            lambda_weight=tracker_kwargs.pop("lambda_weight", 0.25),
            text_sim_thresh=tracker_kwargs.pop("text_sim_thresh", 0.15),
            min_box_area=self.min_box_area,
            mot20=tracker_kwargs.pop("mot20", False),
            **tracker_kwargs,
        )
        self.tracker = self._build_tracker(tracker_type, tracker_args, frame_rate=self.frame_rate)
        self.tracker_type = tracker_type

        # CLIP setup
        self.class_names = [c.strip() for c in self.text_prompt.split(".") if c.strip()] or ["object"]
        self.text_embedding = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_pad = int(tracker_kwargs.pop("clip_pad", 4))

        need_clip = self.tracker_type in ("clip", "smartclip") or referring_mode != "none"
        if need_clip:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            with torch.no_grad():
                tokens = clip.tokenize(self.class_names).to(self.device)
                self.text_embedding = F.normalize(
                    self.clip_model.encode_text(tokens).float(), dim=-1
                ).contiguous()

        # Referring filter
        self.referring_filter = None
        if referring_mode != "none" and self.clip_model is not None:
            self.referring_filter = ReferringDetectionFilter(
                clip_model=self.clip_model,
                clip_preprocess=self.clip_preprocess,
                text_embedding=self.text_embedding,
                mode=referring_mode,
                topk=referring_topk,
                threshold=referring_thresh,
                pad=self.clip_pad,
                device=self.device
            )
            print(f"[Worker] Referring filter: mode={referring_mode}, topk={referring_topk}, thresh={referring_thresh}")

    @staticmethod
    def _build_tracker(tracker_type: str, tracker_args: argparse.Namespace, *, frame_rate: int):
        """Build tracker from registry."""
        if tracker_type not in TRACKER_REGISTRY:
            raise ValueError(f"Unknown tracker: {tracker_type}. Available: {list(TRACKER_REGISTRY.keys())}")
        module_path, class_name = TRACKER_REGISTRY[tracker_type]
        module = importlib.import_module(module_path)
        TrackerCls = getattr(module, class_name)
        return TrackerCls(tracker_args, frame_rate=frame_rate)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess frame for DINO."""
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = self._transform(img)
        if str(self.device).startswith("cuda"):
            tensor = tensor.cuda(non_blocking=True)
        return tensor.half() if self.use_fp16 else tensor

    def predict_detections(self, frame_bgr: np.ndarray, tensor_image: Optional[torch.Tensor],
                          orig_h: int, orig_w: int) -> np.ndarray:
        """Run object detection."""
        if self.detector_kind == "dino":
            with torch.no_grad(), autocast(enabled=self.use_fp16):
                boxes, logits, _ = predict(
                    model=self.dino_model,
                    image=tensor_image,
                    caption=self.text_prompt,
                    box_threshold=self.box_thresh,
                    text_threshold=self.text_thresh,
                )
            return convert_dino_to_xyxy(boxes, logits, orig_w, orig_h)
        else:
            return self.florence.predict(
                frame_bgr=frame_bgr,
                text_prompt=self.text_prompt,
                box_threshold=self.box_thresh
            )

    def update_tracker(self, dets_xyxy: np.ndarray, orig_h: int, orig_w: int):
        """Update tracker with detections."""
        if dets_xyxy.size == 0:
            dets_xyxy = np.empty((0, 5), dtype=np.float32)
        return self.tracker.update(dets_xyxy, [orig_h, orig_w], [orig_h, orig_w])

    def update_tracker_clip(self, dets_xyxy: np.ndarray, frame_bgr: np.ndarray,
                           orig_h: int, orig_w: int):
        """Update CLIP-aware tracker."""
        dets = dets_xyxy if dets_xyxy.size else np.empty((0, 5), dtype=np.float32)
        det_embs = self._compute_detection_embeddings(frame_bgr, dets)
        return self.tracker.update(
            detections=dets,
            detection_embeddings=det_embs,
            img_info=(orig_h, orig_w),
            text_embedding=self.text_embedding,
            class_names=self.class_names,
        )

    def _compute_detection_embeddings(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray) -> List[Optional[torch.Tensor]]:
        """
        Compute CLIP embeddings for detections using both crop and full image context.

        This combines:
        1. Crop embedding: captures object-specific features
        2. Full image embedding: provides spatial context for expressions like "car on the left"

        The two embeddings are averaged to preserve both local and global information.
        """
        if dets_xyxy.size == 0:
            return []

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Compute full image embedding once
        full_img_pil = Image.fromarray(rgb)
        full_img_tensor = self.clip_preprocess(full_img_pil).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            full_img_emb = F.normalize(self.clip_model.encode_image(full_img_tensor), dim=-1).float().cpu().squeeze(0)

        # Compute crop embeddings
        crops = []
        for (x1, y1, x2, y2, _) in dets_xyxy.tolist():
            xi1 = max(0, int(x1) - self.clip_pad)
            yi1 = max(0, int(y1) - self.clip_pad)
            xi2 = min(W, int(x2) + self.clip_pad)
            yi2 = min(H, int(y2) + self.clip_pad)

            if xi2 > xi1 and yi2 > yi1 and (xi2 - xi1) >= 10 and (yi2 - yi1) >= 10:
                crops.append(Image.fromarray(rgb[yi1:yi2, xi1:xi2]))
            else:
                crops.append(None)

        batch = [self.clip_preprocess(c).unsqueeze(0) for c in crops if c is not None]
        if not batch:
            return [None] * len(crops)

        batch_t = torch.cat(batch, 0).to(self.device, non_blocking=True)
        with torch.no_grad():
            crop_embs = F.normalize(self.clip_model.encode_image(batch_t), dim=-1).float().cpu()

        # Combine crop and full image embeddings
        out, j = [], 0
        for c in crops:
            if c is None:
                out.append(None)
            else:
                # Average crop embedding with full image embedding
                # This preserves both object-specific features and spatial context
                combined_emb = F.normalize((crop_embs[j] + full_img_emb) / 2.0, dim=-1)
                out.append(combined_emb)
                j += 1
        return out

    @staticmethod
    def _write_mot_line(fh, frame_id: int, track_id: int, x: float, y: float, w: float, h: float):
        """Write MOTChallenge format line."""
        fh.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    def process_sequence(
        self,
        *,
        seq: str,
        img_folder: str,
        gt_folder: str,
        out_path: str,
        sort_frames: bool = True,
        video_out_path: Optional[str] = None,
    ):
        """
        Process a sequence and generate tracking results.

        Args:
            seq: Sequence name
            img_folder: Root folder containing sequence images
            gt_folder: Root folder containing ground truth
            out_path: Output path for tracking results
            sort_frames: Whether to sort frames by ID
            video_out_path: Optional video output path
        """
        seq_path = os.path.join(img_folder, seq)
        if not os.path.isdir(seq_path):
            raise FileNotFoundError(f"Sequence path not found: {seq_path}")

        # Load GT data
        gt_pandas_data = None
        if self.dataset_type == "mot":
            gt_txt_file = os.path.join(gt_folder, "gt", seq + ".txt")
            if os.path.isfile(gt_txt_file):
                gt_pandas_data = pd.read_csv(
                    gt_txt_file, header=None,
                    names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "x1", "x2", "x3", "x4"],
                    sep=","
                )
                gt_pandas_data.sort_values(by="frame", inplace=True)

        # Get frame files
        frame_files = [f for f in os.listdir(seq_path) if os.path.isfile(os.path.join(seq_path, f))]
        if sort_frames:
            frame_files = sorted(frame_files, key=parse_frame_id)

        # Setup video writer
        video_writer = None
        if self.save_video:
            if video_out_path is None:
                video_out_path = out_path.replace(".txt", ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if self.target_object_ids:
                print(f"[{seq}] Tracking {len(self.target_object_ids)} target object IDs: {sorted(self.target_object_ids)}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w") as f_res:
            for idx, frame_name in enumerate(frame_files):
                frame_id = parse_frame_id(frame_name)
                img = cv2.imread(os.path.join(seq_path, frame_name))
                if img is None:
                    continue
                orig_h, orig_w = img.shape[:2]

                if self.save_video and video_writer is None:
                    video_writer = cv2.VideoWriter(video_out_path, fourcc, self.frame_rate, (orig_w, orig_h))
                    print(f"[{seq}] Saving video to: {video_out_path}")

                # Preprocess
                if self.detector_kind == "dino":
                    tensor = self.preprocess_frame(img)
                    if idx < self.verbose_first_n_frames:
                        _, proc_h, proc_w = tensor.shape
                        print(f"[{seq}] Frame {frame_id}: Original {orig_h}x{orig_w} | "
                              f"Processed {proc_h}x{proc_w} | tracker={type(self.tracker).__name__} | "
                              f"detector={self.detector_kind}")
                else:
                    tensor = None
                    if idx < self.verbose_first_n_frames:
                        print(f"[{seq}] Frame {frame_id}: Original {orig_h}x{orig_w} | "
                              f"Processed n/a (florence2) | tracker={type(self.tracker).__name__} | "
                              f"detector={self.detector_kind}")

                # Detect
                dets = self.predict_detections(img, tensor, orig_h, orig_w)
                if idx < self.verbose_first_n_frames:
                    print(f"[{seq}] Frame {frame_id}: Detected {len(dets)} objects")

                # Filter
                if self.referring_filter is not None:
                    dets_before = len(dets)
                    dets = self.referring_filter.filter(img, dets)
                    if idx < self.verbose_first_n_frames:
                        print(f"[{seq}] Frame {frame_id}: Referring filter {dets_before} → {len(dets)} detections")

                # Track
                if self.tracker_type in ("clip", "smartclip"):
                    tracks = self.update_tracker_clip(dets, img, orig_h, orig_w)
                else:
                    tracks = self.update_tracker(dets, orig_h, orig_w)

                if idx < self.verbose_first_n_frames:
                    print(f"[{seq}] Frame {frame_id}: Tracking {len(tracks)} objects")

                # Write results
                for t in tracks:
                    x, y, w, h = t.tlwh
                    if w * h > self.min_box_area:
                        self._write_mot_line(f_res, frame_id, t.track_id, float(x), float(y), float(w), float(h))

                # Video output
                if self.save_video and video_writer is not None:
                    vis_frame = img.copy()

                    # Draw predicted tracks (green)
                    for t in tracks:
                        x, y, w, h = t.tlwh
                        if w * h > self.min_box_area:
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + w), int(y + h)
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(vis_frame, f"ID:{t.track_id}", (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw GT boxes if enabled
                    if self.show_gt_boxes:
                        if self.dataset_type == "referkitti":
                            # ReferKITTI: YOLO-style labels (only show target objects for referring expressions)
                            if self.referkitti_data_root:
                                label_path = os.path.join(
                                    self.referkitti_data_root, "KITTI", "training",
                                    "labels_with_ids", "image_02", seq, f"{frame_id:06d}.txt"
                                )
                                vis_frame = draw_referkitti_gt_boxes(vis_frame, label_path, target_ids=self.target_object_ids)
                        elif gt_pandas_data is not None:
                            # MOT: pandas format
                            gt_frame_data = gt_pandas_data[gt_pandas_data["frame"] == frame_id]
                            for _, row in gt_frame_data.iterrows():
                                x1 = int(row["bb_left"])
                                y1 = int(row["bb_top"])
                                w = int(row["bb_width"])
                                h = int(row["bb_height"])
                                x2, y2 = x1 + w, y1 + h
                                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(vis_frame, f"GT ID:{int(row['id'])}", (x1 + w, y1 - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    video_writer.write(vis_frame)

        print(f"[{seq}] Saved results to: {out_path}")

        if video_writer is not None:
            video_writer.release()
            print(f"[{seq}] Saved video to: {video_out_path}")

        if self.referring_filter is not None:
            stats = self.referring_filter.get_stats()
            print(f"[{seq}] Referring filter: {stats['total_in']} → {stats['total_out']} "
                  f"({stats['retention_rate']*100:.1f}% retention)")

    def process_many(self, *, seqs: Iterable[str], img_folder: str, res_folder: str,
                    gt_folder: str, suffix: str = ".txt"):
        """Process multiple sequences."""
        os.makedirs(res_folder, exist_ok=True)
        for seq in seqs:
            out_path = os.path.join(res_folder, f"{seq}{suffix}")
            self.process_sequence(seq=seq, img_folder=img_folder, gt_folder=gt_folder, out_path=out_path)


# ============================
# CLI
# ============================
if __name__ == "__main__":
    import glob as _glob
    from datetime import datetime

    def list_sequences(img_root: str):
        return sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])

    def collect_sequences(args) -> List[str]:
        seqs = set()
        if args.seq:
            seqs.update(args.seq)
        if args.seq_file:
            with open(args.seq_file) as fh:
                seqs.update(line.strip() for line in fh if line.strip() and not line.startswith("#"))
        if args.seq_glob:
            for pat in args.seq_glob:
                for p in _glob.glob(os.path.join(args.img_folder, pat)):
                    if os.path.isdir(p):
                        seqs.add(os.path.basename(p))
        if args.all or not seqs:
            seqs.update(list_sequences(args.img_folder))
        return sorted(seqs)

    def resolve_single_out(seq: str, out_arg: Optional[str], out_dir: Optional[str], timestamp: bool) -> str:
        if out_arg and out_arg.lower().endswith(".txt"):
            os.makedirs(os.path.dirname(out_arg), exist_ok=True)
            return out_arg
        root = out_arg or out_dir or "outputs"
        if timestamp:
            root = os.path.join(root, datetime.now().strftime("%Y-%m-%d_%H%M"))
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"{seq}.txt")

    def dispatch_multi_gpu(seqs: List[str], args, tracker_kv: dict):
        devices = [d.strip() for d in (args.devices or "0").split(",") if d.strip()]
        jobs = max(1, int(args.jobs))
        procs = []

        root = args.out_dir or "outputs"
        if args.timestamp:
            root = os.path.join(root, datetime.now().strftime("%Y-%m-%d_%H%M"))
        os.makedirs(root, exist_ok=True)

        this_script = os.path.abspath(__file__)
        for i, seq in enumerate(seqs):
            gpu_id = devices[i % len(devices)]
            out_path = os.path.join(root, f"{seq}.txt")

            if args.save_video:
                video_folder = root.replace("/results", "/videos").replace("\\results", "\\videos")
                if "results" not in root:
                    video_folder = os.path.join(os.path.dirname(root), "videos")
                os.makedirs(video_folder, exist_ok=True)
                video_path = os.path.join(video_folder, f"{seq}.mp4")

            cmd = [
                sys.executable, "-u", this_script,
                "--seq", seq,
                "--img_folder", args.img_folder,
                "--out", out_path,
                "--tracker", args.tracker,
                "--box_thresh", str(args.box_thresh),
                "--text_thresh", str(args.text_thresh),
                "--track_thresh", str(args.track_thresh),
                "--match_thresh", str(args.match_thresh),
                "--track_buffer", str(args.track_buffer),
                "--text_prompt", args.text_prompt,
                "--detector", args.detector,
                "--config", args.config,
                "--weights", args.weights,
                "--min_box_area", str(args.min_box_area),
                "--frame_rate", str(args.frame_rate),
                "--dataset_type", args.dataset_type,
                "--child"
            ]
            if args.use_fp16:
                cmd.append("--use_fp16")
            if args.save_video:
                cmd.append("--save_video")
                cmd.extend(["--video_out", video_path])
            if args.show_gt_boxes:
                cmd.append("--show_gt_boxes")
            for k, v in (tracker_kv or {}).items():
                cmd.extend(["--tracker_kv", f"{k}={v}"])

            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "PYTHONWARNINGS": "ignore::UserWarning,ignore::FutureWarning",
                "TRANSFORMERS_VERBOSITY": "error",
                "MPLBACKEND": "Agg",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            })

            p = subprocess.Popen(cmd, env=env)
            procs.append(p)

            if len(procs) >= jobs:
                procs[0].wait()
                procs = procs[1:]

        for p in procs:
            p.wait()

    parser = argparse.ArgumentParser(description="Clean worker for GroundingDINO + Tracker evaluation")

    # Sequence selection
    parser.add_argument("--seq", nargs="*", help="Sequence names")
    parser.add_argument("--seq_file", type=str, help="File with sequence names")
    parser.add_argument("--seq_glob", action="append", help="Glob patterns for sequences")
    parser.add_argument("--all", action="store_true", help="Process all sequences")
    parser.add_argument("--img_folder", required=True, type=str, help="Image root folder")

    # Output
    parser.add_argument("--out", type=str, help="Output file/folder")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to output")
    parser.add_argument("--video_out", type=str, help="Video output path")
    parser.add_argument("--save_video", action="store_true", help="Save tracking video")
    parser.add_argument("--show_gt_boxes", action="store_true", help="Show GT boxes in video")

    # Dataset
    parser.add_argument("--dataset_type", choices=["mot", "referkitti"], default="mot", help="Dataset type")

    # Detector
    parser.add_argument("--detector", choices=["dino", "florence2"], default="dino", help="Detector type")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Model config path")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Model weights path")
    parser.add_argument("--box_thresh", type=float, default=0.35, help="Box threshold")
    parser.add_argument("--text_thresh", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--text_prompt", type=str, default=DEFAULT_TEXT_PROMPT, help="Text prompt")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16")

    # Tracker
    parser.add_argument("--tracker", default="bytetrack", choices=list(TRACKER_REGISTRY.keys()), help="Tracker type")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="Track threshold")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="Match threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="Track buffer")
    parser.add_argument("--tracker_kv", action="append", help="Tracker key=value args")

    # CLIP
    parser.add_argument("--lambda_weight", type=float, default=0.25, help="CLIP fusion weight")
    parser.add_argument("--low_thresh", type=float, default=0.1, help="Low detection threshold")
    parser.add_argument("--text_sim_thresh", type=float, default=0.0, help="Min CLIP text similarity")
    parser.add_argument("--use_clip_in_high", action="store_true", help="Use CLIP in high conf stage")
    parser.add_argument("--use_clip_in_low", action="store_true", help="Use CLIP in low conf stage")
    parser.add_argument("--use_clip_in_unconf", action="store_true", help="Use CLIP in unconf stage")

    # Misc
    parser.add_argument("--min_box_area", type=int, default=DEFAULT_MIN_BOX_AREA, help="Min box area")
    parser.add_argument("--frame_rate", type=int, default=DEFAULT_FRAME_RATE, help="Frame rate")

    # Multi-GPU
    parser.add_argument("--devices", type=str, help="GPU IDs (comma-separated)")
    parser.add_argument("--jobs", type=int, default=1, help="Max concurrent jobs")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    tracker_kwargs = {
        "track_thresh": args.track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
        "lambda_weight": args.lambda_weight,
        "low_thresh": args.low_thresh,
        "text_sim_thresh": args.text_sim_thresh,
        "use_clip_in_high": args.use_clip_in_high,
        "use_clip_in_low": args.use_clip_in_low,
        "use_clip_in_unconf": args.use_clip_in_unconf,
    }
    tracker_kwargs.update(parse_kv_list(args.tracker_kv))

    # Child mode
    if args.child:
        if not args.seq or len(args.seq) != 1 or not args.out:
            raise SystemExit("Child mode needs exactly one --seq and --out")
        worker = Worker(
            tracker_type=args.tracker,
            tracker_kwargs=tracker_kwargs,
            box_thresh=args.box_thresh,
            text_thresh=args.text_thresh,
            use_fp16=args.use_fp16,
            text_prompt=args.text_prompt,
            detector=args.detector,
            frame_rate=args.frame_rate,
            save_video=args.save_video,
            show_gt_boxes=args.show_gt_boxes,
            dataset_type=args.dataset_type,
            min_box_area=args.min_box_area,
            config_path=args.config,
            weights_path=args.weights,
        )
        out_path = args.out if args.out.lower().endswith(".txt") else os.path.join(args.out, f"{args.seq[0]}.txt")
        video_out = args.video_out if hasattr(args, 'video_out') and args.video_out else None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        worker.process_sequence(
            seq=args.seq[0],
            img_folder=args.img_folder,
            gt_folder=os.path.join(args.img_folder, ".."),
            out_path=out_path,
            video_out_path=video_out
        )
        raise SystemExit(0)

    # Parent mode
    seqs = collect_sequences(args)

    if args.devices and len(seqs) > 1:
        dispatch_multi_gpu(seqs, args, tracker_kwargs)
    else:
        worker = Worker(
            tracker_type=args.tracker,
            tracker_kwargs=tracker_kwargs,
            box_thresh=args.box_thresh,
            text_thresh=args.text_thresh,
            use_fp16=args.use_fp16,
            text_prompt=args.text_prompt,
            detector=args.detector,
            frame_rate=args.frame_rate,
            save_video=args.save_video,
            show_gt_boxes=args.show_gt_boxes,
            dataset_type=args.dataset_type,
            min_box_area=args.min_box_area,
            config_path=args.config,
            weights_path=args.weights,
        )

        if len(seqs) == 1:
            out_path = resolve_single_out(seqs[0], args.out, args.out_dir, args.timestamp)
            worker.process_sequence(
                seq=seqs[0],
                img_folder=args.img_folder,
                gt_folder=os.path.join(args.img_folder, ".."),
                out_path=out_path
            )
        else:
            root = args.out_dir
            if args.timestamp:
                root = os.path.join(root, datetime.now().strftime("%Y-%m-%d_%H%M"))
            os.makedirs(root, exist_ok=True)

            if worker.save_video:
                video_folder = root.replace("/results", "/videos").replace("\\results", "\\videos")
                if "results" not in root:
                    video_folder = os.path.join(os.path.dirname(root), "videos")
                os.makedirs(video_folder, exist_ok=True)

            for s in seqs:
                out_path = os.path.join(root, f"{s}.txt")
                video_path = os.path.join(video_folder, f"{s}.mp4") if worker.save_video else None
                worker.process_sequence(
                    seq=s,
                    img_folder=args.img_folder,
                    out_path=out_path,
                    gt_folder=os.path.join(args.img_folder, ".."),
                    video_out_path=video_path
                )
