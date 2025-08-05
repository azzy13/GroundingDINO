import os
import cv2
import torch
import numpy as np
from PIL import Image
import clip
from groundingdino.util.inference import load_model, predict
from tracker.tracker_w_clip import BYTETracker  # custom version
from torch.cuda.amp import autocast
import torchvision.transforms as T
import torch.nn.functional as F

CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "car. pedestrian."

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TrackerArgs:
    def __init__(self, track_thresh=0.41, track_buffer=198, match_thresh=0.87):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = 10.0
        self.min_box_area = 100
        self.mot20 = False

def preprocess_frame(frame, size, use_fp16=False):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).cuda()
    tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear')[0]
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

def draw_tracks(image, tracks):
    for t in tracks:
        x, y, w, h = map(int, t.tlwh)
        tid = t.track_id
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"ID: {tid}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    return image

def main(video_path, output_path, box_threshold=0.4, text_threshold=0.5, use_fp16=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device).eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    text_tokens = clip.tokenize([TEXT_PROMPT]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker_args = TrackerArgs()
    tracker = BYTETracker(tracker_args, frame_rate=fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor = preprocess_frame(frame, size=(height, width), use_fp16=use_fp16)
        with torch.no_grad(), autocast(enabled=use_fp16):
            boxes, logits, _ = predict(model, image_tensor, TEXT_PROMPT, box_threshold, text_threshold)

        detections = convert_dino_boxes_to_detections(boxes, logits, width, height)
        detection_embeddings = []

        for det in detections:
            x1, y1, x2, y2, score = map(int, det)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                detection_embeddings.append(None)
                continue

            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            clip_input = clip_preprocess(crop_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = clip_model.encode_image(clip_input)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                detection_embeddings.append(emb.squeeze(0).cuda())  # keep on GPU

        tracks = tracker.update(detections, detection_embeddings, (height, width), text_embedding)
        frame_annotated = draw_tracks(frame.copy(), tracks)
        out.write(frame_annotated)

    cap.release()
    out.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    main(args.video, args.output, use_fp16=args.fp16)
