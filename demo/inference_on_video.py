import os
import cv2
import torch
import argparse
import numpy as np
import time
from PIL import Image
import torchvision.transforms as T
from groundingdino.util.inference import load_model, predict, annotate
from torch.cuda.amp import autocast

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT = "trucks only"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame, use_fp16=False):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    image_tensor = transform(image_pil).cuda()
    if use_fp16:
        image_tensor = image_tensor.half()
    return image_pil, image_tensor

def main(video_path, output_path, use_fp16=False):
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    model = model.cuda().eval()

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (width, height))

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
                    boxes, logits, phrases = predict(
                        model=model,
                        image=image_tensor,
                        caption=TEXT_PROMPT,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD
                    )
            else:
                boxes, logits, phrases = predict(
                    model=model,
                    image=image_tensor,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
                )

        torch.cuda.synchronize()
        infer_time = time.time() - start_infer
        total_infer_time += infer_time

        annotated = annotate(image_source=np.array(image_source), boxes=boxes, logits=logits, phrases=phrases)
        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        print(f"✅ Frame {frame_idx} - Inference: {infer_time:.3f}s", end="\r")

    cap.release()
    out.release()

    wall_clock = time.time() - start_wall
    avg_infer = total_infer_time / frame_idx if frame_idx else 0
    eff_fps = frame_idx / total_infer_time if total_infer_time > 0 else 0

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
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to save output video")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 for faster inference")
    args = parser.parse_args()

    main(args.video, args.output, args.fp16)
