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
import torch.nn.functional as F

# === Configurable constants ===
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "car. truck. van. person. pedestrian."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
TRACK_THRESH = 0.5
TRACK_BUFFER = 100
MATCH_THRESH = 0.6
MIN_BOX_AREA = 10
FRAME_RATE = 10

# --- No more static TARGET_SIZE!

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
    # Get frame’s native size for dynamic processing
    h, w = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).cuda()
    tensor = letterbox(tensor, size=(h, w))  # Now using original image size!
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

def combine_gt_local(label_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    for gt_file in glob.glob(os.path.join(label_folder, "*.txt")):
        seq = os.path.splitext(os.path.basename(gt_file))[0]
        out_path = os.path.join(out_folder, f"{seq}.txt")
        with open(gt_file) as f_in, open(out_path, 'w') as f_out:
            for line in f_in:
                parts = line.split()
                frame = int(parts[0]); tid=int(parts[1]); cls=parts[2]
                if cls not in ("Car","Van","Truck","Pedestrian","Person"): continue
                x1,y1,x2,y2 = map(float, parts[6:10])
                w,h = x2-x1, y2-y1
                f_out.write(f"{frame},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

def run_inference_local(img_folder, res_folder, use_fp16=False):
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
        with open(out_file, 'w') as f_res:
            for frame_name in sorted(os.listdir(seq_path)):
                frame_id = int(os.path.splitext(frame_name)[0])
                img = cv2.imread(os.path.join(seq_path, frame_name))
                orig_h, orig_w = img.shape[:2]
                tensor = preprocess_frame(img, use_fp16)  # Uses original image size!
                _, proc_h, proc_w = tensor.shape

                if frame_id < 5:
                    print(f"Frame {frame_id}: Original size={orig_h}x{orig_w}, Processed size={proc_h}x{proc_w}")

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
    p.add_argument('--images',default="/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/image_02")
    p.add_argument('--labels', default="/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/label_02")
    p.add_argument('--fp16', action='store_true')
    args = p.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    base_outdir = "outputs"
    run_outdir = os.path.join(base_outdir, timestamp)
    os.makedirs(run_outdir, exist_ok=True)
    out_gt = os.path.join(run_outdir, 'gt_local')
    out_res = os.path.join(run_outdir, 'inference_results')

    print(f"\nAll outputs for this run will be saved in: {run_outdir}\n")

    combine_gt_local(args.labels, out_gt)
    run_inference_local(args.images, out_res, args.fp16)
    eval_all(out_gt, out_res)
