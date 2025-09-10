#!/usr/bin/env python3
import os
import sys
import cv2
import time
import glob
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

# Import your reusable runner
from eval.worker import Worker

# ===== Defaults (you can override via CLI) =====
DEFAULT_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
DEFAULT_WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
DEFAULT_TEXT_PROMPT = "red car ."

DEFAULT_BOX_THRESHOLD   = 0.42
DEFAULT_TEXT_THRESHOLD  = 0.60
DEFAULT_TRACK_THRESH    = 0.41
DEFAULT_MATCH_THRESH    = 0.87
DEFAULT_TRACK_BUFFER    = 180
DEFAULT_MIN_BOX_AREA    = 10
DEFAULT_LAMBDA_WEIGHT   = 0.25
DEFAULT_TEXT_SIM_THRESH = 0.25

# ===== Helpers =====
def extract_frames(video_path: str, frames_dir: str) -> Tuple[int, float, int, int]:
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        cv2.imwrite(os.path.join(frames_dir, f"{n}.png"), frame)
    cap.release()
    if n == 0:
        raise RuntimeError("No frames extracted (empty video?)")

    return n, fps, W, H


def read_mot_results(mot_txt_path: str) -> DefaultDict[int, List[Tuple[int, float, float, float, float]]]:
    per_frame: DefaultDict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    if not os.path.isfile(mot_txt_path):
        return per_frame

    with open(mot_txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            f  = int(float(parts[0]))
            tid = int(float(parts[1]))
            x   = float(parts[2])
            y   = float(parts[3])
            w   = float(parts[4])
            h   = float(parts[5])
            per_frame[f].append((tid, x, y, w, h))
    return per_frame


def draw_tracks(img, tracks_for_frame: List[Tuple[int, float, float, float, float]]) -> None:
    for (tid, x, y, w, h) in tracks_for_frame:
        xi, yi, wi, hi = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (xi, yi), (xi + wi, yi + hi), (0, 255, 0), 2)
        cv2.putText(img, f"ID:{tid}", (xi, max(0, yi - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def write_tracked_video(frames_dir: str, mot_txt_path: str, out_path: str, fps: float, size: Tuple[int, int]) -> None:
    W, H = size
    per_frame = read_mot_results(mot_txt_path)
    frame_files = sorted(
        (Path(p) for p in glob.glob(os.path.join(frames_dir, "*.png"))),
        key=lambda p: int(p.stem)
    )

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for: {out_path}")

    for p in frame_files:
        f = int(p.stem)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        draw_tracks(img, per_frame.get(f, []))
        writer.write(img)

    writer.release()


def main():
    ap = argparse.ArgumentParser(description="Video → frames → Worker(process_sequence) → MOT → rendered video.")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--output", required=True, help="Path to output mp4 with drawn tracks")
    ap.add_argument("--workdir", default=None, help="Optional temp working dir; if unset a sibling folder is made next to output")
    ap.add_argument("--keep-frames", action="store_true", help="Keep extracted frames and MOT results")

    # Choose tracker
    ap.add_argument("--tracker", choices=["bytetrack","clip"], default="clip",
                    help="Which tracker backend to use (Worker registry).")

    # Model + text
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH)
    ap.add_argument("--text-prompt", default=DEFAULT_TEXT_PROMPT)

    # Thresholds (same semantics as worker)
    ap.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    ap.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    ap.add_argument("--track-threshold", type=float, default=DEFAULT_TRACK_THRESH)
    ap.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESH)
    ap.add_argument("--track-buffer", type=int, default=DEFAULT_TRACK_BUFFER)
    ap.add_argument("--min-box-area", type=int, default=DEFAULT_MIN_BOX_AREA)

    # CLIP-only knobs (ignored for bytetrack)
    ap.add_argument("--lambda-weight", type=float, default=DEFAULT_LAMBDA_WEIGHT)
    ap.add_argument("--text-sim-thresh", type=float, default=DEFAULT_TEXT_SIM_THRESH)
    ap.add_argument("--use-clip-in-high", type=str, default="true", help="true/false (clip only)")
    ap.add_argument("--use-clip-in-low", type=str, default="true", help="true/false (clip only)")
    ap.add_argument("--use-clip-in-unconf", type=str, default="true", help="true/false (clip only)")

    # Precision
    ap.add_argument("--fp16", action="store_true")

    args = ap.parse_args()

    video_path = os.path.abspath(args.video)
    output_path = os.path.abspath(args.output)

    # Workspace
    if args.workdir is None:
        root = os.path.splitext(output_path)[0] + "_work"
    else:
        root = os.path.abspath(args.workdir)
    frames_root = os.path.join(root, "frames")
    seq = "0000"
    seq_dir = os.path.join(frames_root, seq)
    os.makedirs(seq_dir, exist_ok=True)

    # 1) Extract frames
    t0 = time.time()
    num_frames, fps, W, H = extract_frames(video_path, seq_dir)
    t1 = time.time()
    print(f"[extract] {num_frames} frames at {fps:.2f} FPS, size {W}x{H} in {(t1 - t0):.2f}s")

    # 2) Build Worker and run once on this sequence
    tracker_kwargs = dict(
        track_thresh=args.track_threshold,
        track_buffer=args.track_buffer,
        match_thresh=args.match_threshold,
    )
    if args.tracker == "clip":
        tracker_kwargs.update(dict(
            lambda_weight=args.lambda_weight,
            text_sim_thresh=args.text_sim_thresh,
            use_clip_in_high=(str(args.use_clip_in_high).lower() == "true"),
            use_clip_in_low=(str(args.use_clip_in_low).lower() == "true"),
            use_clip_in_unconf=(str(args.use_clip_in_unconf).lower() == "true"),
        ))

    worker = Worker(
        tracker_type=args.tracker,
        tracker_kwargs=tracker_kwargs,
        box_thresh=args.box_threshold,
        text_thresh=args.text_threshold,
        use_fp16=args.fp16,
        text_prompt=args.text_prompt,
        frame_rate=int(round(fps)),
        min_box_area=args.min_box_area,
        config_path=args.config,
        weights_path=args.weights,
    )

    mot_dir = os.path.join(root, "mot")
    os.makedirs(mot_dir, exist_ok=True)
    mot_file = os.path.join(mot_dir, f"{seq}.txt")

    t2 = time.time()
    worker.process_sequence(seq=seq, img_folder=frames_root, out_path=mot_file, sort_frames=True)
    t3 = time.time()
    print(f"[worker] processed {num_frames} frames in {(t3 - t2):.2f}s "
          f"({num_frames / max(1e-6, (t3 - t2)):.2f} FPS)")

    # 3) Render annotated video
    t4 = time.time()
    write_tracked_video(seq_dir, mot_file, output_path, fps=fps, size=(W, H))
    t5 = time.time()
    print(f"[render] wrote {output_path} in {(t5 - t4):.2f}s")

    # 4) Cleanup
    if not args.keep_frames:
        try:
            shutil.rmtree(root)
        except Exception:
            pass

    print("[done] total time: {:.2f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
