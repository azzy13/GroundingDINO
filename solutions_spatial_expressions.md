# Solutions for Spatial Expression Tracking

## Problem Statement

Expression: **"black cars on the right"**

**Current Behavior**:
- Detects BOTH left and right black cars
- Increasing CLIP threshold drops everything
- Spatial information is lost

**Root Cause**: CLIP crop embeddings cannot distinguish spatial location. A crop of "car on left" looks identical to "car on right".

---

## Solution 1: Trust GroundingDINO's Spatial Understanding ⭐ RECOMMENDED

### Why This Works

GroundingDINO **already understands spatial expressions** through cross-modal attention between text and image features. It's designed for referring expression grounding.

### Implementation

**Disable CLIP referring filter** and let GroundingDINO do the spatial filtering:

```bash
python3 eval/eval_referkitti.py \
  --data_root dataset/referkitti/ \
  --devices 0,1 --jobs 2 --fp16 --frame_rate 10 \
  --box_threshold 0.4 --text_threshold 0.8 \
  --track_thresh 0.45 --match_thresh 0.85 \
  --weights /isis/home/hasana3/vlmtest/GroundingDINO/weights/swinb_light_visdrone_ft_best.pth \
  --save_video --show_gt_boxes \
  --tracker clip --use_clip_in_low --use_clip_in_unconf \
  --lambda_weight 0.25 \
  --referring_mode none \  # ◄── Disable CLIP post-filter
  --text_sim_thresh 0.0    # ◄── Disable text similarity gating in tracker
```

### How It Works

```
Expression: "black cars on the right"
    ↓
GroundingDINO (worker_clean.py:457)
    - Uses transformer cross-attention
    - Understands spatial relationships
    - Outputs: Only cars on the right side
    ↓
CLIP Tracker (optional)
    - Uses CLIP for appearance-based re-identification
    - text_sim_thresh=0.0 disables semantic gating
    - CLIP only used for tracking association, NOT filtering
    ↓
Result: Only right-side cars tracked
```

### Tuning GroundingDINO Thresholds

If GroundingDINO detects too many objects:

```bash
# Increase text_threshold to require stronger text-region alignment
--text_threshold 0.9   # Default 0.8, higher = stricter spatial matching

# Increase box_threshold for higher confidence
--box_threshold 0.45   # Default 0.4
```

---

## Solution 2: Position-Augmented CLIP Embeddings

If you need CLIP filtering for some reason, augment embeddings with explicit position information.

### Implementation

Modify `_compute_detection_embeddings()` to include position features:

```python
def _compute_detection_embeddings_with_position(
    self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray
) -> List[Optional[torch.Tensor]]:
    """Compute CLIP embeddings augmented with position information."""
    if dets_xyxy.size == 0:
        return []

    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Full image embedding
    full_img_pil = Image.fromarray(rgb)
    full_img_tensor = self.clip_preprocess(full_img_pil).unsqueeze(0).to(self.device)
    with torch.no_grad():
        full_img_emb = F.normalize(
            self.clip_model.encode_image(full_img_tensor), dim=-1
        ).float().cpu().squeeze(0)

    # Crop embeddings + position encoding
    crops = []
    position_features = []

    for (x1, y1, x2, y2, _) in dets_xyxy.tolist():
        # Compute normalized position features
        cx = (x1 + x2) / (2 * W)  # Center X (0-1)
        cy = (y1 + y2) / (2 * H)  # Center Y (0-1)
        w_norm = (x2 - x1) / W    # Width (0-1)
        h_norm = (y2 - y1) / H    # Height (0-1)

        # Add position encoding: [cx, cy, w, h, left_right, top_bottom]
        left_right = 1.0 if cx > 0.5 else -1.0   # Right=1, Left=-1
        top_bottom = 1.0 if cy > 0.5 else -1.0   # Bottom=1, Top=-1

        pos_feat = torch.tensor(
            [cx, cy, w_norm, h_norm, left_right, top_bottom],
            dtype=torch.float32
        )
        position_features.append(pos_feat)

        # Crop image
        xi1 = max(0, int(x1) - self.clip_pad)
        yi1 = max(0, int(y1) - self.clip_pad)
        xi2 = min(W, int(x2) + self.clip_pad)
        yi2 = min(H, int(y2) + self.clip_pad)

        if xi2 > xi1 and yi2 > yi1:
            crops.append(Image.fromarray(rgb[yi1:yi2, xi1:xi2]))
        else:
            crops.append(None)

    # Compute crop embeddings
    batch = [self.clip_preprocess(c).unsqueeze(0) for c in crops if c is not None]
    if not batch:
        return [None] * len(crops)

    batch_t = torch.cat(batch, 0).to(self.device)
    with torch.no_grad():
        crop_embs = F.normalize(
            self.clip_model.encode_image(batch_t), dim=-1
        ).float().cpu()

    # Combine: crop + full image + position
    out, j = [], 0
    for i, c in enumerate(crops):
        if c is None:
            out.append(None)
        else:
            # Weighted combination favoring spatial context
            combined_emb = F.normalize(
                0.3 * crop_embs[j] + 0.7 * full_img_emb,  # More weight on full image
                dim=-1
            )

            # Append position features (6-dim) to embedding (512-dim)
            # Result: 518-dim vector
            augmented_emb = torch.cat([combined_emb, position_features[i]], dim=0)
            out.append(augmented_emb)
            j += 1

    return out
```

**Pros**: Position explicitly encoded
**Cons**: Requires modifying tracker to handle 518-dim embeddings (breaks compatibility)

---

## Solution 3: Spatial-Aware Text Similarity (QUICK FIX)

Compute CLIP similarity using **full image embedding only** for spatial expressions.

### Detection Logic

```python
def _compute_similarities_spatial_aware(
    self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray
) -> np.ndarray:
    """
    For spatial expressions, use full image context instead of crops.
    """
    if dets_xyxy.size == 0:
        return np.array([])

    # Check if expression contains spatial keywords
    spatial_keywords = ['left', 'right', 'top', 'bottom', 'center',
                       'front', 'back', 'middle', 'side']
    is_spatial_expr = any(kw in self.text_embedding_text.lower()
                          for kw in spatial_keywords)

    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if is_spatial_expr:
        # For spatial expressions: use full image + position masking
        similarities = []
        for (x1, y1, x2, y2, _) in dets_xyxy:
            # Create masked image (zero out non-detection regions)
            mask = np.zeros((H, W, 3), dtype=np.uint8)
            mask[int(y1):int(y2), int(x1):int(x2)] = rgb[int(y1):int(y2), int(x1):int(x2)]

            # Encode masked full image
            masked_pil = Image.fromarray(mask)
            masked_tensor = self.clip_preprocess(masked_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                masked_emb = F.normalize(
                    self.clip_model.encode_image(masked_tensor), dim=-1
                ).float()

            # Compute similarity with text
            sim = F.cosine_similarity(
                self.text_embedding, masked_emb, dim=-1
            ).item()
            similarities.append(sim)

        return np.array(similarities)
    else:
        # For appearance-only expressions: use crop embeddings
        # ... (original implementation)
        pass
```

**Pros**: Preserves spatial context
**Cons**: Slower (one CLIP forward pass per detection)

---

## Solution 4: Increase Full Image Weight in Fusion

Modify line 539 in `worker_clean.py`:

```python
# Current (50/50):
combined_emb = F.normalize((crop_embs[j] + full_img_emb) / 2.0, dim=-1)

# Proposed (30/70 - favor spatial context):
combined_emb = F.normalize(0.3 * crop_embs[j] + 0.7 * full_img_emb, dim=-1)
```

**Pros**: Simple one-line change
**Cons**: Still limited spatial understanding, may hurt appearance matching

---

## Solution 5: Use Explicit Position Filter Post-CLIP

Add explicit bounding box filtering based on text keywords.

```python
def filter_by_spatial_keywords(
    self, dets_xyxy: np.ndarray, frame_width: int, text: str
) -> np.ndarray:
    """Filter detections by spatial keywords in text."""
    if dets_xyxy.size == 0:
        return dets_xyxy

    text_lower = text.lower()

    # Compute detection centers
    centers_x = (dets_xyxy[:, 0] + dets_xyxy[:, 2]) / 2

    # Filter by keywords
    if 'right' in text_lower or 'rightmost' in text_lower:
        # Keep only detections in right half
        mask = centers_x > (frame_width * 0.5)
        return dets_xyxy[mask]
    elif 'left' in text_lower or 'leftmost' in text_lower:
        # Keep only detections in left half
        mask = centers_x < (frame_width * 0.5)
        return dets_xyxy[mask]
    elif 'center' in text_lower or 'middle' in text_lower:
        # Keep only detections in center third
        mask = (centers_x > frame_width * 0.33) & (centers_x < frame_width * 0.67)
        return dets_xyxy[mask]

    return dets_xyxy  # No spatial keyword, return all
```

**Pros**: Simple and effective for basic spatial expressions
**Cons**: Brittle keyword matching, doesn't handle complex expressions

---

## Recommended Configuration

### For ReferKITTI with Spatial Expressions:

```bash
python3 eval/eval_referkitti.py \
  --data_root dataset/referkitti/ \
  --devices 0,1 --jobs 2 --fp16 --frame_rate 10 \
  --box_threshold 0.35 \
  --text_threshold 0.85 \        # ◄── Higher for better spatial matching
  --track_thresh 0.45 \
  --match_thresh 0.85 \
  --weights /path/to/weights.pth \
  --save_video --show_gt_boxes \
  --tracker clip \
  --use_clip_in_low \
  --use_clip_in_unconf \
  --lambda_weight 0.25 \
  --referring_mode none \         # ◄── Trust GroundingDINO
  --text_sim_thresh 0.0           # ◄── No CLIP semantic gating
```

### If You MUST Use Referring Filter:

```bash
# Lower topk to be more selective
--referring_mode topk \
--referring_topk 1 \              # ◄── Keep only most similar detection per frame
--text_sim_thresh 0.0             # ◄── Don't use text-sim gating in tracker
```

---

## Understanding the Trade-offs

| Approach | Spatial Understanding | Appearance Matching | Speed | Complexity |
|----------|---------------------|-------------------|-------|------------|
| **Solution 1: Trust GroundingDINO** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| Solution 2: Position Features | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Solution 3: Full Image Matching | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Solution 4: Adjust Fusion Weight | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| Solution 5: Keyword Filter | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

---

## Why CLIP Struggles with Spatial Expressions

### CLIP's Training

CLIP was trained on **image-text pairs** with descriptions like:
- ✅ "A black car"
- ✅ "A luxury sedan"
- ❌ "The car on the right" (rare in training data)

### Spatial Understanding Requires:

1. **Global context**: Full image understanding
2. **Relational reasoning**: Understanding "left of", "next to", etc.
3. **Position encoding**: Explicit coordinate information

### GroundingDINO's Advantage:

GroundingDINO uses **cross-modal attention** between:
- Visual features at spatial locations
- Text tokens (including spatial words)

This allows direct grounding of "right" to image regions, which CLIP crops cannot achieve.

---

## Recommended Workflow

1. **Start with Solution 1** (trust GroundingDINO)
2. **Tune `text_threshold`** to balance precision/recall
3. **Only use CLIP tracker for re-identification** across frames (not filtering)
4. **If GroundingDINO fails**, try Solution 5 (keyword filter) as post-processing

This leverages each component's strengths:
- **GroundingDINO**: Spatial grounding
- **CLIP**: Appearance-based tracking association
