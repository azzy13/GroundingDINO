# Detection Flow and Text Prompt in Worker

## Quick Answer

**Where**: Detection happens in `worker_clean.py:454` in the `predict_detections()` method

**Prompt**: Uses `self.text_prompt` which comes from `expr["text"]` (the referring expression)

**Example**: For expression "black cars on the right", GroundingDINO receives exactly that text as the detection prompt.

---

## Complete Detection Flow

### Step-by-Step Trace

```
┌─────────────────────────────────────────────────────────────┐
│ 1. eval_referkitti.py - Load Expression                    │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │ Line 185-305:                     │
        │ load_expressions_for_sequence()   │
        │                                   │
        │ Loads: expression/0001/*.json     │
        │                                   │
        │ Returns:                          │
        │ {                                 │
        │   "expr_id": 0,                   │
        │   "text": "black cars on right",  │
        │   "obj_ids": [5, 8]               │
        │ }                                 │
        └───────────────┬───────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. eval_referkitti.py:617 - Create Worker                  │
│                                                             │
│ worker = Worker(                                            │
│     text_prompt=expr["text"],  ◄── "black cars on right"   │
│     box_thresh=0.4,                                         │
│     text_thresh=0.8,                                        │
│     ...                                                     │
│ )                                                           │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. worker_clean.py:352 - Store Text Prompt                 │
│                                                             │
│ def __init__(self, text_prompt="car. pedestrian.", ...):   │
│     self.text_prompt = text_prompt                          │
│     # Now self.text_prompt = "black cars on right"         │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. worker_clean.py:645 - process_sequence()                │
│                                                             │
│ For each frame in video:                                   │
│   1. Load image                                             │
│   2. Preprocess (line 616)                                 │
│   3. Detect (line 630) ◄── DETECTION HAPPENS HERE          │
│   4. Filter (line 635-639) - optional                      │
│   5. Track (line 642-645)                                  │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. worker_clean.py:449-467 - predict_detections()          │
│                                                             │
│ def predict_detections(self, frame_bgr, tensor_image,      │
│                        orig_h, orig_w):                     │
│     if self.detector_kind == "dino":                        │
│         with torch.no_grad():                               │
│             boxes, logits, _ = predict(                     │
│                 model=self.dino_model,                      │
│                 image=tensor_image,                         │
│                 caption=self.text_prompt, ◄── PROMPT USED! │
│                 box_threshold=self.box_thresh,              │
│                 text_threshold=self.text_thresh,            │
│             )                                               │
│         return convert_dino_to_xyxy(boxes, logits, ...)    │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. GroundingDINO Model (groundingdino/util/inference.py)   │
│                                                             │
│ predict() function:                                         │
│   - Encodes image with Swin Transformer backbone           │
│   - Encodes text with BERT encoder                         │
│   - Cross-attention between image & text features          │
│   - Predicts boxes aligned with text phrases               │
│                                                             │
│ Returns:                                                    │
│   - boxes: Bounding boxes [N, 4] (normalized)              │
│   - logits: Confidence scores [N]                          │
│   - phrases: Matched text phrases                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Details

### Where Detection Happens

**File**: `eval/worker_clean.py`
**Method**: `predict_detections()` at line 449
**Called from**: `process_sequence()` at line 630

```python
# Line 630 in process_sequence():
dets = self.predict_detections(img, tensor, orig_h, orig_w)

# Lines 449-467 - predict_detections() implementation:
def predict_detections(self, frame_bgr: np.ndarray, tensor_image: Optional[torch.Tensor],
                      orig_h: int, orig_w: int) -> np.ndarray:
    """Run object detection."""
    if self.detector_kind == "dino":
        with torch.no_grad(), autocast(enabled=self.use_fp16):
            boxes, logits, _ = predict(
                model=self.dino_model,          # Pre-loaded GroundingDINO model
                image=tensor_image,              # Preprocessed frame tensor
                caption=self.text_prompt,        # ◄── Expression text!
                box_threshold=self.box_thresh,   # e.g., 0.4
                text_threshold=self.text_thresh, # e.g., 0.8
            )
        return convert_dino_to_xyxy(boxes, logits, orig_w, orig_h)
    else:
        # Florence-2 alternative
        return self.florence.predict(
            frame_bgr=frame_bgr,
            text_prompt=self.text_prompt,
            box_threshold=self.box_thresh
        )
```

---

## Text Prompt Source

### Default Prompt

**File**: `eval/worker_clean.py:39`

```python
DEFAULT_TEXT_PROMPT = "car. pedestrian."
```

### ReferKITTI Override

**File**: `eval/eval_referkitti.py:617`

```python
worker = Worker(
    text_prompt=expr["text"],  # Referring expression from JSON
    ...
)
```

### Example Expression JSONs

**Format 1**: ReferKITTI style (`expression/0001/000000.json`)

```json
{
  "label": {
    "1": [11],
    "2": [11],
    "3": [11]
  },
  "ignore": {},
  "video_name": "0001",
  "sentence": "black cars on the right"
}
```

**Format 2**: Generic style

```json
{
  "expression": "the leftmost car",
  "obj_ids": [3, 7]
}
```

---

## How GroundingDINO Uses the Text Prompt

### 1. Text Encoding (BERT)

```python
# In GroundingDINO internals:
text = "black cars on the right"

# Tokenize
tokens = tokenizer(text)  # ["black", "cars", "on", "the", "right"]

# Encode with BERT
text_features = bert_encoder(tokens)  # [num_tokens, 768]
```

### 2. Image Encoding (Swin Transformer)

```python
# Extract multi-scale image features
image_features = swin_backbone(image)  # [H/32, W/32, C]
```

### 3. Cross-Modal Attention

```python
# Align text tokens with image regions
for layer in cross_attention_layers:
    # Image attends to text
    image_features = image_to_text_attention(image_features, text_features)

    # Text attends to image
    text_features = text_to_image_attention(text_features, image_features)
```

**Key**: This allows GroundingDINO to understand:
- **"black"** → match image regions with dark colors
- **"cars"** → match car-shaped objects
- **"on the right"** → focus on right side of image

### 4. Box Prediction

```python
# Predict boxes from fused features
boxes, scores = box_predictor(image_features)

# Filter by thresholds
boxes = boxes[scores > box_threshold]  # e.g., 0.4

# Filter by text alignment
aligned = text_alignment_score(boxes, text_features)
boxes = boxes[aligned > text_threshold]  # e.g., 0.8
```

---

## Detection Parameters

### box_threshold (default: 0.35)

**What it does**: Filters detections by objectness confidence

**Effect**:
- **Lower** (e.g., 0.3): More detections, higher recall, more false positives
- **Higher** (e.g., 0.5): Fewer detections, higher precision, may miss objects

**Example**:
```python
# Detection scores: [0.9, 0.45, 0.32, 0.88]
# box_threshold=0.4 → Keep: [0.9, 0.45, 0.88]
```

### text_threshold (default: 0.25)

**What it does**: Filters detections by text-image alignment score

**Critical for spatial expressions!**

**Effect**:
- **Lower** (e.g., 0.25): Loose text matching, may include unrelated objects
- **Higher** (e.g., 0.85): Strict text matching, better spatial filtering

**Example**:
```python
# Text: "black cars on the right"
# Detection 1 (black car, right side): text_score=0.92 ✓
# Detection 2 (black car, left side): text_score=0.65 ✗ (with thresh=0.8)
# Detection 3 (white truck, right): text_score=0.45 ✗
```

**For spatial expressions, use higher text_threshold (0.8-0.9)!**

---

## Complete Per-Frame Detection Pipeline

```python
# In process_sequence() - Lines 603-648

for frame_file in sorted(frame_files):
    # 1. Load frame
    img = cv2.imread(frame_path)  # [H, W, 3] BGR
    orig_h, orig_w = img.shape[:2]

    # 2. Preprocess for GroundingDINO
    tensor = self.preprocess_frame(img)  # Normalize & resize

    # 3. DETECT with text prompt
    dets = self.predict_detections(img, tensor, orig_h, orig_w)
    # Returns: np.ndarray [N, 5] = [x1, y1, x2, y2, score]
    # Example: [[100, 50, 250, 180, 0.87], [300, 60, 450, 200, 0.72]]

    # 4. Optional: CLIP referring filter (if referring_mode != "none")
    if self.referring_filter is not None:
        dets = self.referring_filter.filter(img, dets)
        # Further filters by CLIP text-image similarity
        # WARNING: This breaks spatial understanding!

    # 5. Track detections
    if self.tracker_type in ("clip", "smartclip"):
        tracks = self.update_tracker_clip(dets, img, orig_h, orig_w)
    else:
        tracks = self.update_tracker(dets, orig_h, orig_w)

    # 6. Write results
    for t in tracks:
        write_mot_line(f_res, frame_id, t.track_id, *t.tlwh)
```

---

## Key Insights

### 1. **One Text Prompt per Expression**

Each referring expression gets its own Worker instance with a specific text prompt:

```python
# Expression 0: "black cars on the right"
worker_0 = Worker(text_prompt="black cars on the right")

# Expression 1: "white truck in the front"
worker_1 = Worker(text_prompt="white truck in the front")
```

### 2. **Same Prompt for All Frames**

The text prompt is **constant** across all frames in a sequence:
- Frame 1: detect("black cars on the right")
- Frame 2: detect("black cars on the right")
- ...

### 3. **GroundingDINO is Spatial-Aware**

GroundingDINO's cross-attention mechanism understands:
- Spatial terms: "left", "right", "top", "bottom", "center"
- Relational terms: "next to", "behind", "in front of"
- Ordinal terms: "leftmost", "rightmost", "first", "last"

**This is why you should trust GroundingDINO and disable CLIP post-filtering!**

### 4. **Two-Stage Filtering**

```
Stage 1: GroundingDINO Detection
  ↓
  Understands: appearance + spatial + relational
  ↓
  Filters by: box_threshold, text_threshold
  ↓
Stage 2: CLIP Referring Filter (OPTIONAL - often harmful!)
  ↓
  Understands: appearance only (NO spatial!)
  ↓
  Filters by: topk or similarity threshold
  ↓
Result: Spatial information may be lost!
```

---

## Recommended Settings for Spatial Expressions

### Trust GroundingDINO

```bash
python3 eval/eval_referkitti.py \
  --data_root dataset/referkitti/ \
  --box_threshold 0.35 \         # Objectness confidence
  --text_threshold 0.85 \        # ◄── High for spatial filtering!
  --referring_mode none \        # ◄── Disable CLIP filter
  --text_sim_thresh 0.0          # ◄── Disable tracker gating
```

### If Using CLIP Filter (NOT recommended for spatial)

```bash
--referring_mode topk \
--referring_topk 1 \              # Only keep top-1 per frame
--text_threshold 0.85             # Still use high DINO threshold
```

---

## Debugging Detection Output

### Add Verbose Output

The Worker already has verbose logging for first 5 frames:

```python
# Line 632
if idx < self.verbose_first_n_frames:
    print(f"[{seq}] Frame {frame_id}: Detected {len(dets)} objects")

# Line 638-639
if self.referring_filter is not None and idx < self.verbose_first_n_frames:
    print(f"[{seq}] Frame {frame_id}: Referring filter {dets_before} → {len(dets)} detections")
```

### Increase Verbosity

Modify line 345 in worker_clean.py:

```python
# Current:
verbose_first_n_frames: int = 5

# More verbose:
verbose_first_n_frames: int = 20
```

### Check Detection Quality

Add this after line 630 in `process_sequence()`:

```python
dets = self.predict_detections(img, tensor, orig_h, orig_w)

# DEBUG: Print detection details
if idx < 10:  # First 10 frames
    print(f"\nFrame {frame_id} detections:")
    for i, (x1, y1, x2, y2, score) in enumerate(dets):
        cx = (x1 + x2) / (2 * orig_w)  # Normalized center X
        position = "RIGHT" if cx > 0.5 else "LEFT"
        print(f"  Det {i}: score={score:.2f}, center_x={cx:.2f}, position={position}")
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Where** | `worker_clean.py:454` in `predict_detections()` |
| **Function** | `predict()` from `groundingdino.util.inference` |
| **Prompt Source** | `expr["text"]` from expression JSON |
| **Prompt Format** | Natural language (e.g., "black cars on the right") |
| **Spatial Understanding** | ✅ GroundingDINO: YES, ❌ CLIP: NO |
| **Key Parameters** | `box_threshold`, `text_threshold` |
| **Recommendation** | Use `text_threshold=0.85+` for spatial expressions |

GroundingDINO receives the full referring expression and understands spatial relationships through cross-modal attention. The CLIP referring filter should be disabled (`referring_mode=none`) to preserve this spatial understanding!
