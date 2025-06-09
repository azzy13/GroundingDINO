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

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT = "trucks. cars."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame, use_fp16=False):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    image_tensor = transform(image_pil).cuda()
    if use_fp16:
        image_tensor = image_tensor.half()
    return image_pil, image_tensor

def convert_dino_boxes_to_detections(boxes, logits, image_w, image_h):
    dets = []
    for box, logit in zip(boxes, logits):
        cx, cy, w, h = box
        score = float(logit)

        x1 = (cx - w / 2) * image_w
        y1 = (cy - h / 2) * image_h
        x2 = (cx + w / 2) * image_w
        y2 = (cy + h / 2) * image_h

        # Filter clearly invalid boxes
        if w > 0.8 or h > 0.8 or w <= 0 or h <= 0:
            continue

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_w - 1, x2), min(image_h - 1, y2)

        dets.append([x1, y1, x2, y2, score])
    return np.array(dets)



class TrackerArgs:
    track_thresh = 0.3
    track_buffer = 60
    match_thresh = 0.7
    aspect_ratio_thresh = 10.0
    min_box_area = 100
    mot20 = False

def draw_tracks(image, tracks):
    for t in tracks:
        x, y, w, h = map(int, t.tlwh)
        tid = t.track_id
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"ID: {tid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return image

def main(video_path, output_path, use_fp16=False):
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    model = model.cuda().eval()

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker_args = TrackerArgs()
    tracker = BYTETracker(tracker_args, frame_rate=fps)

    with open("track_results.txt", "w") as results_txt:
        frame_idx = 0
        total_infer_time = 0.0
        start_wall = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_source, image_tensor = preprocess_frame(frame, use_fp16)

            torch.cuda.synchronize()
            start_infer = time.time()

            with torch.no_grad():
                if use_fp16:
                    with autocast():
                        boxes, logits, _ = predict(
                            model=model,
                            image=image_tensor,
                            caption=TEXT_PROMPT,
                            box_threshold=BOX_THRESHOLD,
                            text_threshold=TEXT_THRESHOLD
                        )
                else:
                    boxes, logits, _ = predict(
                        model=model,
                        image=image_tensor,
                        caption=TEXT_PROMPT,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD
                    )

            torch.cuda.synchronize()
            infer_time = time.time() - start_infer
            total_infer_time += infer_time
            
            # print("DINO boxes:", boxes)
            # print("DINO logits:", logits)
            # print("Width:", width, "Height:", height)

            detections = convert_dino_boxes_to_detections(boxes, logits, width, height)
            if detections is None or len(detections) == 0:
                online_targets = []
            else:
                online_targets = tracker.update(detections, [height, width], [height, width])


            for t in online_targets:
                x, y, w, h = t.tlwh
                tid = t.track_id
                if w * h > tracker_args.min_box_area:
                    results_txt.write(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

            annotated_frame = draw_tracks(np.array(image_source), online_targets)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            frame_idx += 1
            print(f"✅ Frame {frame_idx} - Inference: {infer_time:.3f}s", end="\r")

    cap.release()
    out.release()

    wall_clock = time.time() - start_wall
    avg_infer = total_infer_time / frame_idx
    eff_fps = frame_idx / total_infer_time

    print(f"\n🎬 Video saved to: {output_path}")
    print("===================================")
    print(f"Total Frames: {frame_idx}")
    print(f"Total Inference Time: {total_infer_time:.2f}s")
    print(f"Average Inference Time per Frame: {avg_infer:.4f}s")
    print(f"Effective FPS: {eff_fps:.2f}")
    print(f"Total Wall Clock Time: {wall_clock:.2f}s")
    print("===================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for inference")
    args = parser.parse_args()

    main(args.video, args.output, args.fp16)