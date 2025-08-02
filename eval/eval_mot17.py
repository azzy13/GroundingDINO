import os
import cv2
import glob
import argparse
import motmetrics as mm
import numpy as np
import torch
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from groundingdino.util.inference import load_model, predict
from tracker.byte_tracker import BYTETracker
from torch.cuda.amp import autocast
import torch.nn.functional as F
from datetime import datetime

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "person"
MIN_BOX_AREA = 10
FRAME_RATE = 30  # Usually MOT17 is 30 FPS

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

def convert_dino_boxes_to_detections(boxes, logits, W, H):
    dets = []
    for box, logit in zip(boxes, logits):
        cx, cy, w, h = box
        score = float(logit)
        x1 = (cx - w/2) * W
        y1 = (cy - h/2) * H
        x2 = (cx + w/2) * W
        y2 = (cy + h/2) * H
        if w <= 0 or h <= 0:
            continue
        dets.append([max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2), score])
    return np.array(dets) if dets else np.empty((0,5))

def run_inference_mot17(img_dir, out_file, box_thresh, text_thresh, track_thresh, match_thresh, track_buffer, use_fp16=False):
    model = load_model(CONFIG_PATH, WEIGHTS_PATH).cuda().eval()
    tracker = BYTETracker(
        argparse.Namespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            aspect_ratio_thresh=10.0,
            min_box_area=MIN_BOX_AREA,
            mot20=False
        ),
        frame_rate=FRAME_RATE
    )
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    with open(out_file, 'w') as f_res:
        for idx, img_path in enumerate(img_files):
            frame_id = idx + 1  # MOT starts at 1
            img = cv2.imread(img_path)
            orig_h, orig_w = img.shape[:2]
            tensor = preprocess_frame(img, use_fp16)
            with torch.no_grad(), autocast(enabled=use_fp16):
                boxes, logits, _ = predict(
                    model=model, image=tensor,
                    caption=TEXT_PROMPT,
                    box_threshold=box_thresh,
                    text_threshold=text_thresh
                )
            dets = convert_dino_boxes_to_detections(boxes, logits, orig_w, orig_h)
            tracks = tracker.update(dets, [orig_h,orig_w], [orig_h,orig_w]) if dets.size else []
            for t in tracks:
                x, y, w, h = t.tlwh; tid = t.track_id
                if w*h > MIN_BOX_AREA:
                    # MOTChallenge format: frame,id,bb_left,bb_top,bb_width,bb_height,score,-1,-1,-1
                    f_res.write(
                        f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        print(f"Saved tracking results to {out_file}")

def eval_all(gt_folder, res_folder):
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "MOT17-??/gt/gt.txt")))
    res_files = [os.path.join(res_folder, f"{os.path.basename(os.path.dirname(gt)).split('-')[1]}.txt") for gt in gt_files]

    all_metrics = [
        'num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr',
        'precision', 'recall', 'num_switches',
        'mostly_tracked', 'mostly_lost', 'num_fragmentations',
        'num_false_positives', 'num_misses', 'num_objects'
    ]
    mh = mm.metrics.create()
    all_summaries = []

    for gt_f, res_f in zip(gt_files, res_files):
        seq = os.path.basename(os.path.dirname(gt_f))
        gt = mm.io.loadtxt(gt_f, fmt='mot15-2D', min_confidence=1)
        res = mm.io.loadtxt(res_f, fmt='mot15-2D')
        acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=0.5)
        summary = mh.compute(acc, metrics=all_metrics, name=seq)
        all_summaries.append(summary)
        print(f"\n===== Sequence: {seq} =====")
        print(mm.io.render_summary(
            summary,
            namemap=mm.io.motchallenge_metric_names,
            formatters=mh.formatters,
        ))
    df_all = pd.concat(all_summaries)
    avg_row = df_all.mean(numeric_only=True)
    avg_row.name = 'AVG'
    df_all = df_all.append(avg_row)
    print("\n====== AVERAGE ACROSS SEQUENCES ======")
    print(df_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mot_root', default="/isis/home/hasana3/vlmtest/GroundingDINO/dataset/MOT17/train", help="Path to MOT17/train/")
    parser.add_argument('--box_threshold', type=float, default=0.42)
    parser.add_argument('--text_threshold', type=float, default=0.5)
    parser.add_argument('--track_thresh', type=float, default=0.41)
    parser.add_argument('--match_thresh', type=float, default=0.87)
    parser.add_argument('--track_buffer', type=int, default=200)
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    # Where to save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    outdir = args.outdir or os.path.join("outputs", f"mot17_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    # Run inference for each FRCNN sequence in MOT17/train
    for seq_folder in sorted(glob.glob(os.path.join(args.mot_root, "MOT17-*-FRCNN"))):
        seq_name = os.path.basename(seq_folder)
        img_dir = os.path.join(seq_folder, "img1")
        out_file = os.path.join(outdir, f"{seq_name}.txt")
        print(f"Processing {seq_name} ...")
        run_inference_mot17(
            img_dir, out_file,
            args.box_threshold, args.text_threshold,
            args.track_thresh, args.match_thresh,
            args.track_buffer, args.fp16
        )

    # Evaluate on all sequences
    gt_root = args.mot_root
    eval_all(gt_root, outdir)
