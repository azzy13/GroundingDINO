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
from groundingdino.util.inference import load_model, predict
from tracker.byte_tracker import BYTETracker
from torch.cuda.amp import autocast

# === Configurable constants ===
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "car. truck. van. person. pedestrian."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
TRACK_THRESH = 0.4
TRACK_BUFFER = 100
MATCH_THRESH = 0.6
MIN_BOX_AREA = 10
FRAME_RATE = 30  # VisDrone typical

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame, use_fp16=False):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).cuda()
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

def combine_gt_visdrone(label_folder, out_folder):
    """
    Converts VisDrone MOT val annotations to MOT format used by motmetrics:
    frame_id, track_id, x, y, w, h, 1, -1, -1, -1
    """
    os.makedirs(out_folder, exist_ok=True)
    for ann_file in glob.glob(os.path.join(label_folder, "*.txt")):
        seq = os.path.splitext(os.path.basename(ann_file))[0]
        out_path = os.path.join(out_folder, f"{seq}.txt")
        with open(ann_file) as fin, open(out_path, 'w') as fout:
            for line in fin:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                frame = int(parts[0])
                tid = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                vis = int(parts[8])
                cls = int(parts[7])
                # Filter: only "fully visible" and real targets (person, car, etc.)
                # VisDrone: cls in [0..10], 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, ...
                if tid <= 0 or vis == 0 or cls == 0:
                    continue
                fout.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

def run_inference_visdrone(img_folder, res_folder, use_fp16=False):
    model = load_model(CONFIG_PATH, WEIGHTS_PATH).cuda().eval()
    os.makedirs(res_folder, exist_ok=True)
    for seq in sorted(os.listdir(img_folder)):
        seq_path = os.path.join(img_folder, seq)
        if not os.path.isdir(seq_path):
            continue
        tracker = BYTETracker(
            argparse.Namespace(
                track_thresh=TRACK_THRESH,
                track_buffer=TRACK_BUFFER,
                match_thresh=MATCH_THRESH,
                aspect_ratio_thresh=10.0,
                min_box_area=MIN_BOX_AREA,
                mot20=False
            ),
            frame_rate=FRAME_RATE
        )
        out_file = os.path.join(res_folder, f"{seq}.txt")
        frame_list = sorted(os.listdir(seq_path))
        for fname in frame_list[:3]:
            print("Sample image for debug:", os.path.join(seq_path, fname))
        with open(out_file, 'w') as f_res:
            for frame_name in sorted(os.listdir(seq_path)):
                if not (frame_name.endswith(".jpg") or frame_name.endswith(".png")):
                    continue
                frame_id = int(os.path.splitext(frame_name)[0])
                img = cv2.imread(os.path.join(seq_path, frame_name))
                orig_h, orig_w = img.shape[:2]
                tensor = preprocess_frame(img, use_fp16)
                # Model input is (orig_h, orig_w)
                with torch.no_grad(), autocast(enabled=use_fp16):
                    boxes, logits, _ = predict(
                        model=model, image=tensor,
                        caption=TEXT_PROMPT,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD
                    )
                dets = convert_dino_boxes_to_detections(boxes, logits, orig_w, orig_h)
                tracks = tracker.update(dets, [orig_h,orig_w], [orig_h,orig_w]) if dets.size else []
                for t in tracks:
                    x,y,w,h = t.tlwh; tid = t.track_id
                    if w*h > MIN_BOX_AREA:
                        f_res.write(
                            f"{frame_id},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
        print(f"Saved tracking results for {seq} to {out_file}")

def eval_all(gt_folder, res_folder):
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.txt")))
    res_files= sorted(glob.glob(os.path.join(res_folder,"*.txt")))
    motas = []
    for gt_f, res_f in zip(gt_files, res_files):
        seq = os.path.basename(gt_f)[:-4]
        gt= mm.io.loadtxt(gt_f, fmt='mot15-2D', min_confidence=1)
        res= mm.io.loadtxt(res_f,fmt='mot15-2D')
        acc= mm.utils.compare_to_groundtruth(gt,res,'iou',distth=0.5)
        mh= mm.metrics.create()
        summary= mh.compute(acc, metrics=['num_frames','mota','motp','idf1','num_switches'], name=seq)
        print(mm.io.render_summary(summary, namemap=mm.io.motchallenge_metric_names))
        motas.append(summary.loc[seq,'mota'])
    print(f"\nAverage MOTA: {np.mean(motas):.3f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--fp16', action='store_true')
    args = p.parse_args()

    # --- Make output subfolder with timestamp ---
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    base_outdir = "outputs"
    run_outdir = os.path.join(base_outdir, f"visdrone_{timestamp}")
    os.makedirs(run_outdir, exist_ok=True)
    out_gt = os.path.join(run_outdir, 'gt_local')
    out_res = os.path.join(run_outdir, 'inference_results')

    print(f"\nAll outputs for this run will be saved in: {run_outdir}\n")

    combine_gt_visdrone(args.labels, out_gt)
    run_inference_visdrone(args.images, out_res, args.fp16)
    eval_all(out_gt, out_res)
