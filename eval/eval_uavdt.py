#!/usr/bin/env python3
"""
UAVDT MOT evaluation using worker.py + motmetrics.

Usage:
  python eval/eval_uavdt.py \
    --data_root dataset/UAV \
    --img_root UAV-benchmark-M \
    --gt_root UAV-benchmark-MOTD_v1.0/GT \
    --text_prompt "car. truck. bus."
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

from compute_metrics import MotMetricsEvaluator

WORKER_PY = Path(__file__).resolve().parent / "worker.py"


def collect_sequences(img_root: Path):
    return sorted(
        d.name for d in img_root.iterdir()
        if d.is_dir() and d.name.upper().startswith("M")
    )


def copy_uavdt_gt(gt_root: Path, sequences, out_gt: Path):
    out_gt.mkdir(parents=True, exist_ok=True)
    copied = 0
    for seq in sequences:
        src = gt_root / f"{seq}_gt.txt"
        if src.is_file():
            dst = out_gt / f"{seq}.txt"
            shutil.copy(src, dst)
            copied += 1
    print(f"\n📋 Copied {copied} GT files → {out_gt}")


def link_images(img_root: Path, sequences, temp_images: Path):
    temp_images.mkdir(parents=True, exist_ok=True)
    print(f"\n🔗 Creating image symlinks for {len(sequences)} sequences...")
    for seq in sequences:
        src = img_root / seq
        dst = temp_images / seq
        if not src.is_dir():
            print(f"   ✗ Missing: {src}")
            continue
        if dst.exists():
            continue
        try:
            os.symlink(src, dst, target_is_directory=True)
            print(f"   ✓ {seq}")
        except OSError:
            shutil.copytree(src, dst)
            print(f"   ✓ {seq} (copied)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Base UAVDT path containing UAV-benchmark-M and GT folder.")
    ap.add_argument("--img_root", default="UAV-benchmark-M",
                    help="Relative path from data_root to images root.")
    ap.add_argument("--gt_root", default="UAV-benchmark-MOTD_v1.0/GT",
                    help="Relative path from data_root to GT root with Mxxxx_gt.txt.")
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.10)
    ap.add_argument("--track_thresh", type=float, default=0.30)
    ap.add_argument("--match_thresh", type=float, default=0.85)
    ap.add_argument("--track_buffer", type=int, default=80)
    ap.add_argument("--tracker", choices=["bytetrack", "clip"], default="bytetrack")
    ap.add_argument("--detector", choices=["dino", "florence2"], default="dino")
    ap.add_argument("--text_prompt", type=str, default="car. truck. bus.")
    ap.add_argument("--config", type=str,
                    default="groundingdino/config/GroundingDINO_SwinB_cfg.py")
    ap.add_argument("--weights", type=str,
                    default="weights/groundingdino_swinb_cogcoor.pth")
    ap.add_argument("--min_box_area", type=int, default=10)
    ap.add_argument("--frame_rate", type=int, default=25)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--devices", type=str, default="0")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    img_root = (data_root / args.img_root).resolve()
    gt_root = (data_root / args.gt_root).resolve()

    if not img_root.is_dir():
        sys.exit(f"No image root dir: {img_root}")
    if not gt_root.is_dir():
        sys.exit(f"No GT root dir: {gt_root}")

    seqs = collect_sequences(img_root)
    if not seqs:
        sys.exit(f"No Mxxxx sequences found under {img_root}")

    # Outputs
    if args.outdir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        run_outdir = Path("outputs") / f"uavdt_test_{ts}"
    else:
        run_outdir = Path(args.outdir)
    out_gt = run_outdir / "gt"
    out_res = run_outdir / "results"
    temp_images = run_outdir / "images"
    out_res.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("UAVDT Evaluation - TEST")
    print(f"Output directory: {run_outdir}")
    print("=" * 60 + "\n")

    copy_uavdt_gt(gt_root, seqs, out_gt)
    link_images(img_root, seqs, temp_images)

    # Run worker
    cmd = [
        sys.executable, "-u", str(WORKER_PY),
        "--img_folder", str(temp_images),
        "--all",
        "--out_dir", str(out_res),
        "--tracker", args.tracker,
        "--box_thresh", str(args.box_threshold),
        "--text_thresh", str(args.text_threshold),
        "--track_thresh", str(args.track_thresh),
        "--match_thresh", str(args.match_thresh),
        "--track_buffer", str(args.track_buffer),
        "--text_prompt", args.text_prompt,
        "--config", args.config,
        "--weights", args.weights,
        "--min_box_area", str(args.min_box_area),
        "--frame_rate", str(args.frame_rate),
        "--devices", args.devices,
        "--jobs", str(args.jobs),
        "--detector", args.detector,
    ]
    if args.fp16:
        cmd.append("--use_fp16")
    if args.save_video:
        cmd.append("--save_video")

    print("\n🚀 Running tracking on sequences...\n")
    print(" \\\n  ".join(cmd))
    print()

    subprocess.check_call(cmd)

    print("\n" + "=" * 60)
    print("📊 Evaluating Results")
    print("=" * 60 + "\n")

    evaluator = MotMetricsEvaluator(distth=0.5, fmt="mot15-2D")
    df = evaluator.evaluate(str(out_gt), str(out_res), verbose=True)

    if df is not None and not df.empty:
        mota = float(df.loc["AVG"].get("mota", df["mota"].mean()))
        idf1 = float(df.loc["AVG"].get("idf1", df["idf1"].mean()))
        print("\n" + "=" * 60)
        print(f"📈 Summary: MOTA={mota*100:.2f}% | IDF1={idf1*100:.2f}%")
        print("=" * 60 + "\n")
        print(f"OPTUNA:MOTA={mota:.6f} IDF1={idf1:.6f}")
    else:
        print("⚠ No matching GT/RES files for evaluation.")

    print(f"\n✅ Complete! Results saved to: {run_outdir}")


if __name__ == "__main__":
    main()
