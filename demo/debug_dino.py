import cv2
import numpy as np
import os
from datetime import datetime

# --- Config ---
img_dir = '/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/image_02/0018'
gt_file = '/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/label_02/0018.txt'
track_file = '/isis/home/hasana3/vlmtest/GroundingDINO/outputs/2025-06-30_0317/inference_results/0018.txt'

# --- KITTI GT Parser ---
def parse_kitti_txt(file_path, frame_id, valid_classes=('Car', 'Van', 'Truck', 'Pedestrian', 'Person')):
    boxes = []
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            if int(parts[0]) != frame_id:
                continue
            label = parts[2]
            if label not in valid_classes:
                continue
            x1, y1, x2, y2 = map(float, parts[6:10])
            tid = parts[1]
            boxes.append((x1, y1, x2 - x1, y2 - y1, tid))
    return boxes

# --- Tracker Output (MOT Format) Parser ---
def parse_mot_txt(file_path, frame_id):
    boxes = []
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            if int(parts[0]) != frame_id:
                continue
            x, y, w, h = map(float, parts[2:6])
            tid = parts[1]
            boxes.append((x, y, w, h, tid))
    return boxes

# --- Visualization ---
def vis_frame(img_path, gt_file, track_file, frame_id, save_dir):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return

    H, W = img.shape[:2]

    # Draw dummy box for sanity check
    cv2.rectangle(img, (10, 10), (200, 200), (0, 255, 0), 2)

    # Ground Truth (green)
    gt_boxes = parse_kitti_txt(gt_file, frame_id)
    for x, y, ww, hh, _ in gt_boxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + ww), int(y + hh)
        if 0 <= x1 < W and 0 <= y1 < H and 0 < x2 <= W and 0 < y2 <= H:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, 'GT', (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)

    # Tracker (red, with ID)
    track_boxes = parse_mot_txt(track_file, frame_id)
    for x, y, ww, hh, tid in track_boxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + ww), int(y + hh)
        if 0 <= x1 < W and 0 <= y1 < H and 0 < x2 <= W and 0 < y2 <= H:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'ID:{tid}', (x1, y2 + 20), 0, 0.7, (0, 0, 255), 2)

    out_path = os.path.join(save_dir, f"frame_{frame_id:06d}.jpg")
    cv2.imwrite(out_path, img)
    print(f"Saved visualized frame to {out_path}")

# --- Create output folder with timestamp ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
base_save_dir = "debug_vis"
save_dir = os.path.join(base_save_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)
print(f"Saving all visualizations to {save_dir}")

# --- Run for a few frames ---
for frame in range(100, 110):   # Check frames 100–109
    img_file = f"{img_dir}/{str(frame).zfill(6)}.png"
    print(f"\nVisualizing frame: {img_file}")
    if not os.path.exists(img_file):
        print(f"File not found: {img_file}")
        continue
    vis_frame(img_file, gt_file, track_file, frame, save_dir)
