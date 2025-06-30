# Step-by-step debugging in a Jupyter Notebook environment

# 1. Import necessary libraries
import os
import glob
import argparse
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

# Load Grounding DINO model
from groundingdino.util.inference import load_model, predict
from tracker.byte_tracker import BYTETracker
from torch.cuda.amp import autocast
import torch.nn.functional as F

# Define constants
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "car. truck. van. person. pedestrian."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
TRACK_THRESH = 0.4
TRACK_BUFFER = 100
MATCH_THRESH = 0.6
MIN_BOX_AREA = 10
TARGET_SIZE = (768, 1280)

# Initialize model
model = load_model(CONFIG_PATH, WEIGHTS_PATH).cuda().eval()

# Preprocessing functions
def letterbox(tensor, size=TARGET_SIZE):
    _, h, w = tensor.shape
    scale = min(size[0] / h, size[1] / w)
    resized = F.interpolate(tensor.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False)[0]
    pad_h = size[0] - resized.shape[1]
    pad_w = size[1] - resized.shape[2]
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return F.pad(resized, (left, right, top, bottom), value=0.0)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame, use_fp16=False):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(img).cuda()
    tensor = letterbox(tensor, size=TARGET_SIZE)
    return tensor.half() if use_fp16 else tensor

# Load a single test frame
frame_path = '/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/image_02/0018/000095.png'
frame = cv2.imread(frame_path)
orig_h, orig_w = frame.shape[:2]

# Visualize the original frame
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Original Frame")
plt.show()

# Preprocess and visualize
processed_tensor = preprocess_frame(frame, use_fp16=False)
processed_img = processed_tensor.cpu().numpy().transpose(1, 2, 0)
processed_img = (processed_img - processed_img.min()) / (processed_img.max() - processed_img.min())

plt.figure(figsize=(12, 6))
plt.imshow(processed_img)
plt.title("Processed (Letterboxed) Frame")
plt.show()

# Run model prediction
with torch.no_grad(), autocast(enabled=False):
    boxes, logits, _ = predict(
        model=model, image=processed_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

print("Detected boxes:", boxes)

# Convert to original image dimensions
def convert_dino_boxes(boxes, logits, W, H):
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
    return np.array(dets)

detections = convert_dino_boxes(boxes, logits, orig_w, orig_h)
print("Detections in original coordinates:", detections)

# Visualize detections on original frame
for det in detections:
    x1, y1, x2, y2, score = det
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Detections Overlayed")
plt.show()
