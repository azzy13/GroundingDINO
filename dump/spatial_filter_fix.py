# Spatial Keyword Filter - Quick Fix for GroundingDINO Spatial Limitations
# Add this to worker_clean.py

import numpy as np

def filter_by_spatial_keywords(dets_xyxy: np.ndarray, text_prompt: str, frame_width: int, frame_height: int) -> np.ndarray:
    """
    Filter detections by explicit spatial keywords in text prompt.

    Args:
        dets_xyxy: [N, 5] array of [x1, y1, x2, y2, score]
        text_prompt: The referring expression text
        frame_width: Image width
        frame_height: Image height

    Returns:
        Filtered detections
    """
    if dets_xyxy.size == 0:
        return dets_xyxy

    text_lower = text_prompt.lower()

    # Compute detection centers
    centers_x = (dets_xyxy[:, 0] + dets_xyxy[:, 2]) / 2
    centers_y = (dets_xyxy[:, 1] + dets_xyxy[:, 3]) / 2

    # Normalize centers
    centers_x_norm = centers_x / frame_width
    centers_y_norm = centers_y / frame_height

    # Initialize mask (all True)
    mask = np.ones(len(dets_xyxy), dtype=bool)

    # Horizontal filtering
    if 'right' in text_lower or 'rightmost' in text_lower or 'right-hand' in text_lower:
        # Keep only detections in right half (or right 60% to be conservative)
        mask &= centers_x_norm > 0.5
        print(f"[Spatial Filter] 'right' keyword → keeping {mask.sum()}/{len(dets_xyxy)} detections")

    elif 'left' in text_lower or 'leftmost' in text_lower or 'left-hand' in text_lower:
        # Keep only detections in left half
        mask &= centers_x_norm < 0.5
        print(f"[Spatial Filter] 'left' keyword → keeping {mask.sum()}/{len(dets_xyxy)} detections")

    elif 'center' in text_lower or 'middle' in text_lower or 'central' in text_lower:
        # Keep only detections in center third
        mask &= (centers_x_norm > 0.33) & (centers_x_norm < 0.67)
        print(f"[Spatial Filter] 'center' keyword → keeping {mask.sum()}/{len(dets_xyxy)} detections")

    # Vertical filtering
    if 'top' in text_lower or 'upper' in text_lower:
        # Keep only detections in top half
        mask &= centers_y_norm < 0.5
        print(f"[Spatial Filter] 'top' keyword → keeping {mask.sum()}/{len(dets_xyxy)} detections")

    elif 'bottom' in text_lower or 'lower' in text_lower:
        # Keep only detections in bottom half
        mask &= centers_y_norm > 0.5
        print(f"[Spatial Filter] 'bottom' keyword → keeping {mask.sum()}/{len(dets_xyxy)} detections")

    # Ordinal filtering (leftmost, rightmost, etc.)
    if 'leftmost' in text_lower and mask.sum() > 1:
        # Among remaining detections, keep only the leftmost
        remaining_dets = dets_xyxy[mask]
        remaining_centers_x = centers_x[mask]
        leftmost_idx = np.argmin(remaining_centers_x)

        # Create new mask keeping only leftmost
        new_mask = np.zeros(len(remaining_dets), dtype=bool)
        new_mask[leftmost_idx] = True

        # Map back to original indices
        original_mask = np.zeros(len(dets_xyxy), dtype=bool)
        original_mask[np.where(mask)[0][new_mask]] = True
        mask = original_mask
        print(f"[Spatial Filter] 'leftmost' → keeping 1 detection")

    elif 'rightmost' in text_lower and mask.sum() > 1:
        # Among remaining detections, keep only the rightmost
        remaining_centers_x = centers_x[mask]
        rightmost_idx = np.argmax(remaining_centers_x)

        new_mask = np.zeros(mask.sum(), dtype=bool)
        new_mask[rightmost_idx] = True

        original_mask = np.zeros(len(dets_xyxy), dtype=bool)
        original_mask[np.where(mask)[0][new_mask]] = True
        mask = original_mask
        print(f"[Spatial Filter] 'rightmost' → keeping 1 detection")

    return dets_xyxy[mask]


# ============================================================
# HOW TO ADD TO worker_clean.py
# ============================================================

"""
1. Add the function above to worker_clean.py (around line 300, before Worker class)

2. In Worker.__init__(), add this parameter:

   Line 351 (after target_object_ids):

   use_spatial_filter: bool = True,  # Enable explicit spatial filtering

3. Store it:

   Line 363 (after self.target_object_ids):

   self.use_spatial_filter = bool(use_spatial_filter)

4. In process_sequence(), add filtering after detection:

   Line 632 (after detection):

   # Detect
   dets = self.predict_detections(img, tensor, orig_h, orig_w)
   if idx < self.verbose_first_n_frames:
       print(f"[{seq}] Frame {frame_id}: Detected {len(dets)} objects")

   # ADD THIS BLOCK:
   # Spatial keyword filtering
   if self.use_spatial_filter and len(dets) > 0:
       dets_before = len(dets)
       dets = filter_by_spatial_keywords(dets, self.text_prompt, orig_w, orig_h)
       if idx < self.verbose_first_n_frames and dets_before != len(dets):
           print(f"[{seq}] Frame {frame_id}: Spatial filter {dets_before} → {len(dets)} detections")

   # Then continue with CLIP referring filter...
   if self.referring_filter is not None:
       ...

5. In eval_referkitti.py, add to Worker creation (line 632):

   worker = Worker(
       ...
       referring_mode=args.referring_mode,
       referring_topk=args.referring_topk,
       referring_thresh=args.referring_thresh,
       use_spatial_filter=True,  # ADD THIS LINE
   )
"""


# ============================================================
# ALTERNATIVE: Quick Test Without Code Changes
# ============================================================

"""
If you want to test this without modifying worker_clean.py,
you can monkey-patch it:

1. Save this file as eval/spatial_filter_fix.py

2. In eval_referkitti.py, add at the top:

   from spatial_filter_fix import filter_by_spatial_keywords
   import eval.worker_clean as worker_module
   worker_module.filter_by_spatial_keywords = filter_by_spatial_keywords

3. Then modify worker_clean.py just line 632 to add:

   dets = self.predict_detections(img, tensor, orig_h, orig_w)
   # Quick spatial filter
   if hasattr(worker_module, 'filter_by_spatial_keywords'):
       dets = worker_module.filter_by_spatial_keywords(dets, self.text_prompt, orig_w, orig_h)
"""
