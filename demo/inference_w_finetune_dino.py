#!/usr/bin/env python3
"""
Inference script using Worker class - same as Docker container but for local use.
"""

import os
import sys
import cv2
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.worker import Worker


def draw_tracks(frame, tracks, min_box_area=100):
    """Draw bounding boxes and IDs on frame."""
    vis = frame.copy()

    # Color palette
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

    for t in tracks:
        if not t.is_activated:
            continue

        x, y, w, h = t.tlwh
        if w * h < min_box_area:
            continue

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        color = tuple(map(int, colors[t.track_id % len(colors)]))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{t.track_id} ({t.score:.2f})"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Video inference with GroundingDINO + ByteTrack")

    # Input/Output
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")

    # Model paths
    parser.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--weights", default="weights/swinb_light_visdrone_ft_best.pth")

    # Detection
    parser.add_argument("--text-prompt", default="car. pedestrian.", help="Detection prompt")
    parser.add_argument("--box-thresh", type=float, default=0.35)
    parser.add_argument("--text-thresh", type=float, default=0.25)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 inference")

    # Tracking
    parser.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "clip"])
    parser.add_argument("--track-thresh", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.85)

    args = parser.parse_args()

    # Initialize worker
    print(f"Loading model: {args.weights}")
    print(f"Text prompt: {args.text_prompt}")
    print(f"FP16: {args.fp16}")

    tracker_kwargs = {
        "track_thresh": args.track_thresh,
        "track_buffer": args.track_buffer,
        "match_thresh": args.match_thresh,
    }

    worker = Worker(
        config_path=args.config,
        weights_path=args.weights,
        text_prompt=args.text_prompt,
        box_thresh=args.box_thresh,
        text_thresh=args.text_thresh,
        use_fp16=args.fp16,
        tracker_type=args.tracker,
        tracker_kwargs=tracker_kwargs,
    )

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

    # Output video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_path = args.output

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Output: {output_path}")
    print("-" * 50)

    frame_idx = 0
    logged_resize = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        tensor = worker.preprocess_frame(frame)

        # Log resize info once
        if not logged_resize:
            _, tensor_h, tensor_w = tensor.shape
            print(f"Resize: {width}x{height} -> {tensor_w}x{tensor_h}")
            logged_resize = True

        # Detect
        dets = worker.predict_detections(frame, tensor, height, width)

        # Track
        if worker.tracker_type == "clip":
            tracks = worker.update_tracker_clip(dets, frame, height, width)
        else:
            tracks = worker.update_tracker(dets, height, width)

        # Draw and write
        vis = draw_tracks(frame, tracks)

        # Add info overlay
        info = f"Frame: {frame_idx} | Detections: {len(dets)} | Tracks: {len(tracks)}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(vis)

        frame_idx += 1
        print(f"Frame {frame_idx}/{total_frames} | Dets: {len(dets)} | Tracks: {len(tracks)}", end="\r")

    cap.release()
    out.release()

    print(f"\nDone! Saved to: {output_path}")


if __name__ == "__main__":
    main()
