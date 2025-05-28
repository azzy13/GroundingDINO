import os
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from groundingdino.util.inference import load_model, predict, annotate
import numpy as np


CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT = "car . truck . bus ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    image_tensor = transform(image_pil).cuda()  # [C, H, W]
    return image_pil, image_tensor  # Note: NOT batched

def main(video_path, output_path):
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_source, image_tensor = preprocess_frame(frame)

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=model,
                image=image_tensor,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

        annotated = annotate(image_source=np.array(image_source), boxes=boxes, logits=logits, phrases=phrases)
        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        print(f"✅ Processed frame {frame_idx}", end="\r")

    cap.release()
    out.release()
    print(f"\n🎬 Saved video to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to save output video")
    args = parser.parse_args()

    main(args.video, args.output)
