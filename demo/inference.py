import os
import cv2
import torch
import argparse
import numpy as np
import time
from PIL import Image
import torchvision.transforms as T
from groundingdino.util.inference import load_model, predict
from tracker.byte_tracker import BYTETracker
from torch.cuda.amp import autocast
import torch.nn.functional as F


CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "car. pedestrian."

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

def convert_dino_boxes_to_detections(boxes, logits, image_w, image_h):
    dets = []
    for box, logit in zip(boxes, logits):
        cx, cy, w, h = box
        score = float(logit)

        x1 = (cx - w / 2) * image_w
        y1 = (cy - h / 2) * image_h
        x2 = (cx + w / 2) * image_w
        y2 = (cy + h / 2) * image_h

        if w > 0.8 or h > 0.8 or w <= 0 or h <= 0:
            continue

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_w - 1, x2), min(image_h - 1, y2)

        dets.append([x1, y1, x2, y2, score])
    return np.array(dets)

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = 10.0
        self.min_box_area = 100
        self.mot20 = False

def draw_tracks(image, tracks):
    for t in tracks:
        x, y, w, h = map(int, t.tlwh)
        tid = t.track_id
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"ID: {tid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return image

def main(video_path, output_path, box_threshold, text_threshold,
         track_thresh, match_thresh, track_buffer, use_fp16=False):
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    model = model.cuda().eval()

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker_args = TrackerArgs(track_thresh, track_buffer, match_thresh)
    tracker = BYTETracker(tracker_args, frame_rate=fps)

    with open("track_results.txt", "w") as results_txt:
        frame_idx = 0
        total_infer_time = 0.0
        start_wall = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_tensor = preprocess_frame(frame, use_fp16)

            torch.cuda.synchronize()
            start_infer = time.time()

            with torch.no_grad(), autocast(enabled=use_fp16):
                boxes, logits, _ = predict(
                    model=model,
                    image=image_tensor,
                    caption=TEXT_PROMPT,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )

            torch.cuda.synchronize()
            infer_time = time.time() - start_infer
            total_infer_time += infer_time

            detections = convert_dino_boxes_to_detections(boxes, logits, width, height)
            online_targets = tracker.update(detections, [height, width], [height, width]) if detections.size else []

            for t in online_targets:
                x, y, w, h = t.tlwh
                tid = t.track_id
                if w * h > tracker_args.min_box_area:
                    results_txt.write(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

            annotated_frame = draw_tracks(frame.copy(), online_targets)
            out.write(annotated_frame)

            frame_idx += 1
            print(f"✅ Frame {frame_idx} - Inference: {infer_time:.3f}s", end="\r")

    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--box-threshold", type=float, default=0.42)
    parser.add_argument("--text-threshold", type=float, default=0.50)
    parser.add_argument("--track-thresh", type=float, default=0.41)
    parser.add_argument("--match-thresh", type=float, default=0.87)
    parser.add_argument("--track-buffer", type=int, default=198)
    args = parser.parse_args()

    main(args.video, args.output, args.box_threshold, args.text_threshold,
         args.track_thresh, args.match_thresh, args.track_buffer, args.fp16)