import os
import cv2
from GroundingDINO.eval.eval_visdrone import TRACK_BUFFER
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

# Dot-separated classes; empty tokens are ignored
TEXT_PROMPT = "cyan car."

# --- Thresholds (eased to ensure track init) ---
BOX_THRESHOLD   = 0.42    # DINO threshold used in predict()
TEXT_THRESHOLD  = 0.60
TRACK_THRESH    = 0.41    # LOWER so new tracks spawn
MATCH_THRESH    = 0.87
LAMBDA_WEIGHT   = 0.25
TEXT_SIM_THRESH = 0.25    # keep disabled until stable (0.25 later if needed)
TRACK_BUFFER    = 180     # frames to keep lost tracks

DEBUG = True  # set False to silence periodic logs

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TrackerArgs:
    def __init__(self,
                 track_thresh=TRACK_THRESH,
                 track_buffer=TRACK_BUFFER,
                 match_thresh=MATCH_THRESH,
                 lambda_weight=LAMBDA_WEIGHT,
                 text_sim_thresh=TEXT_SIM_THRESH):
        self.track_thresh = float(track_thresh)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.lambda_weight = float(lambda_weight)
        self.text_sim_thresh = float(text_sim_thresh)
        self.aspect_ratio_thresh = 10.0
        self.min_box_area = 100
        self.mot20 = False

def preprocess_frame(frame, size, use_fp16=False):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img)  # [3,H,W], float32 CPU
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False
    )[0]  # [3,H,W]
    return tensor.half() if use_fp16 else tensor

def _is_xyxy_pixels(boxes):
    if isinstance(boxes, np.ndarray):
        return boxes.size > 0 and float(np.max(boxes)) > 1.5
    return boxes.numel() > 0 and float(boxes.max().item()) > 1.5

def nms_xyxy(dets, iou_thresh=0.7):
    """
    dets: np.array [N,5] (x1,y1,x2,y2,score), returns kept indices
    """
    if dets.size == 0:
        return []
    x1, y1, x2, y2, s = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def convert_boxes_robust(boxes, logits, image_w, image_h,
                         min_side=8, max_area_frac=0.35, max_aspect_ratio=4.0, nms_iou=0.7):
    """
    Accept (cx,cy,w,h) normalized or (x1,y1,x2,y2) pixels; return np.float32 [N,5].
    Filters: tiny, too-large (area fraction), extreme aspect ratios; then NMS.
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().float().numpy()
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().float().numpy()

    dets = []
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    def _push(x1, y1, x2, y2, score):
        # clip
        x1 = max(0.0, min(float(x1), image_w - 1.0))
        y1 = max(0.0, min(float(y1), image_h - 1.0))
        x2 = max(0.0, min(float(x2), image_w - 1.0))
        y2 = max(0.0, min(float(y2), image_h - 1.0))
        if x2 <= x1 or y2 <= y1:
            return
        w, h = (x2 - x1), (y2 - y1)
        if w < min_side or h < min_side:
            return
        area = w * h
        if area > max_area_frac * (image_w * image_h):
            return
        ar = max(w / (h + 1e-6), h / (w + 1e-6))
        if ar > max_aspect_ratio:
            return
        dets.append([x1, y1, x2, y2, float(score)])

    if _is_xyxy_pixels(boxes):
        for (x1, y1, x2, y2), score in zip(boxes, logits):
            _push(x1, y1, x2, y2, score)
    else:
        for (cx, cy, w, h), score in zip(boxes, logits):
            cx, cy, w, h = map(float, (cx, cy, w, h))
            if w <= 0 or h <= 0:
                continue
            x1 = (cx - w / 2.0) * image_w
            y1 = (cy - h / 2.0) * image_h
            x2 = (cx + w / 2.0) * image_w
            y2 = (cy + h / 2.0) * image_h
            _push(x1, y1, x2, y2, score)

    if len(dets) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    dets = np.asarray(dets, dtype=np.float32)
    keep = nms_xyxy(dets, iou_thresh=nms_iou)
    return dets[keep] if len(keep) else np.zeros((0, 5), dtype=np.float32)

def draw_tracks(image, tracks):
    for t in tracks:
        x, y, w, h = map(int, t.tlwh)
        tid = t.track_id
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"ID:{tid}", (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def parse_classes_from_prompt(prompt: str):
    return [c.strip() for c in prompt.split('.') if c.strip()]

def main(video_path, output_path, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, use_fp16=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GroundingDINO
    model = load_model(CONFIG_PATH, WEIGHTS_PATH).to(device).eval()

    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # Text embeddings (per class, fp32)
    classes = parse_classes_from_prompt(TEXT_PROMPT) or ["object"]
    with torch.no_grad():
        text_tokens = clip.tokenize(classes).to(device)  # [C, L]
        with torch.cuda.amp.autocast(enabled=False):
            text_emb = clip_model.encode_text(text_tokens)      # [C, D] fp32
        text_emb = F.normalize(text_emb.float(), dim=-1).contiguous()

    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker_args = TrackerArgs()
    tracker = BYTETracker(tracker_args, frame_rate=fps)

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fidx += 1

        # DINO
        image_tensor = preprocess_frame(frame, size=(height, width), use_fp16=use_fp16).to(device)
        with torch.no_grad(), autocast(enabled=use_fp16):
            boxes, logits, _ = predict(model, image_tensor, TEXT_PROMPT, box_threshold, text_threshold)

        detections = convert_boxes_robust(boxes, logits, width, height)  # np.float32 [N,5]
        if DEBUG and fidx % 30 == 1:
            print(f"[Frame {fidx}] DINO dets: {len(detections)}")

        # CLIP image embeddings (CPU, fp32, unit norm)
        detection_embeddings = []
        for det in detections:
            x1, y1, x2, y2, _ = det.tolist()
            xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
            crop = frame[yi1:yi2, xi1:xi2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                detection_embeddings.append(None)
                continue
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            clip_input = clip_preprocess(crop_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    emb = clip_model.encode_image(clip_input)   # [1, D] fp32
                emb = F.normalize(emb, dim=-1).squeeze(0).float().cpu()
            detection_embeddings.append(emb)

        # Track
        tracks = tracker.update(
            detections=detections,
            detection_embeddings=detection_embeddings,
            img_info=(height, width),
            text_embedding=text_emb,      # [C,D] fp32
            class_names=classes
        )

        if DEBUG and fidx % 30 == 1:
            print(f"[Frame {fidx}] Tracks active: {len(tracks)}")

        out.write(draw_tracks(frame.copy(), tracks))

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
