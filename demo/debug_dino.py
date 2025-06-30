import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from groundingdino.util.inference import load_model, predict

# —— CONFIG ——
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
TEXT_PROMPT   = "vehicles."  # more generic
BOX_THRESHOLD = 0.15       # much lower
TEXT_THRESHOLD= 0.15      # much lower

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

# —— Load model once ——
model = load_model(CONFIG_PATH, WEIGHTS_PATH).cuda().eval()

# —— Inference on one image ——
img_path = "/isis/home/hasana3/vlmtest/GroundingDINO/dataset/kitti/validation/image_02/0018/000095.png"
img_bgr  = cv2.imread(img_path)
H, W     = img_bgr.shape[:2]

# to tensor
img_pil   = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
img_t     = transform(img_pil).cuda().unsqueeze(0)  # note batch-dim

# predict
with torch.no_grad():
    boxes, logits, _ = predict(
        model=model, image=img_t[0],
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

print(f"Raw boxes: {boxes.shape}, logits: {logits.shape}")

# draw them
for (cx,cy,w,h), score in zip(boxes, logits):
    x1 = int((cx - w/2)*W)
    y1 = int((cy - h/2)*H)
    x2 = int((cx + w/2)*W)
    y2 = int((cy + h/2)*H)
    cv2.rectangle(img_bgr, (x1,y1),(x2,y2),(0,0,255),2)
    cv2.putText(img_bgr,f"{score:.2f}",(x1,y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

cv2.imwrite("debug_dino_vis.png", img_bgr)
print("Wrote debug_dino_vis.png")
