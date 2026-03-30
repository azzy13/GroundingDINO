#!/usr/bin/env python3
"""
Scene graph builder for GroundingDINO + Tracker perception output.

Builds a per-frame scene graph from active tracks:
  - Nodes: one per tracked object, with spatial/appearance attributes
  - Edges: pairwise relations (spatial, size, overlap, appearance similarity)

Usage (from within Worker.process_sequence):
    from scene_graph import SceneGraphBuilder
    sg = SceneGraphBuilder(text_prompt="red sedan")
    sg.update(frame_id, tracks, orig_h, orig_w, frame_bgr=img)
    sg.save_jsonl("output/scenario_001_scene_graphs.jsonl")
"""

from __future__ import annotations

import json
import math
from collections import deque
from typing import List, Optional, Dict, Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Standalone HSV color classifier (subset of worker_clean's patch voting)
# ---------------------------------------------------------------------------

def _classify_hsv_pixel(h_val: float, s_val: float, v_val: float) -> str:
    """Classify one mean-HSV patch value into a color name."""
    if s_val >= 35:
        if h_val < 15 or h_val >= 160:
            return "red"
        elif h_val < 25:
            return "orange"
        elif h_val < 35:
            return "yellow"
        elif h_val < 85:
            return "green"
        else:
            return "blue"
    if v_val < 80:
        return "dark"
    elif v_val > 180:
        return "light"
    return "gray"


def _dominant_color_from_crop(crop_bgr: np.ndarray, grid_size: int = 4):
    """Return (dominant_color_str, votes_dict) using patch-based HSV voting."""
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", {}

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    if h < 2 or w < 2:
        return "unknown", {}

    gs = min(grid_size, h, w)
    ph, pw = max(1, h // gs), max(1, w // gs)
    votes: Dict[str, int] = {}

    for i in range(gs):
        for j in range(gs):
            patch = hsv[i * ph: min((i + 1) * ph, h),
                        j * pw: min((j + 1) * pw, w)]
            if patch.size == 0:
                continue
            mean_px = np.mean(patch.reshape(-1, 3), axis=0)
            color = _classify_hsv_pixel(*mean_px)
            votes[color] = votes.get(color, 0) + 1

    if not votes:
        return "unknown", {}
    dominant = max(votes, key=votes.get)
    return dominant, votes


def _get_track_color(frame_bgr: np.ndarray, x: float, y: float,
                     w: float, h: float) -> tuple[str, dict]:
    """Extract dominant color for a track's bounding box crop."""
    H, W = frame_bgr.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 8 or (y2 - y1) < 8:
        return "unknown", {}
    crop = frame_bgr[y1:y2, x1:x2]
    return _dominant_color_from_crop(crop)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou_tlwh(box1, box2) -> float:
    """IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# SceneGraphBuilder
# ---------------------------------------------------------------------------

class SceneGraphBuilder:
    """
    Builds per-frame scene graphs from tracker output.

    Call .update() once per frame after tracker output.
    Call .save_jsonl() to persist all frames as newline-delimited JSON.
    """

    # Thresholds for spatial relations
    SPATIAL_THRESH = 0.10   # min normalised Δ to call left-of / above etc.
    NEAR_THRESH    = 0.25   # normalised distance ≤ this → "near"
    FAR_THRESH     = 0.50   # normalised distance ≥ this → "far"
    SIZE_RATIO     = 1.30   # area ratio to call larger-than / smaller-than
    OVERLAP_THRESH = 0.05   # IoU ≥ this → "overlapping"
    CLIP_SIM_THRESH = 0.85  # cosine sim ≥ this → "visually-similar"

    def __init__(self, text_prompt: str = ""):
        self.text_prompt = text_prompt
        self.frames: List[Dict[str, Any]] = []
        # Track motion history: track_id -> deque[(cx_norm, cy_norm, area_norm)]
        self._history: Dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_id: int,
        tracks: list,
        img_h: int,
        img_w: int,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Build scene graph for one frame.

        Args:
            frame_id:  integer frame index
            tracks:    list of STrack objects (ByteTrack or CLIPTracker)
            img_h/w:   original image dimensions
            frame_bgr: optional BGR frame for color extraction

        Returns:
            dict with keys: frame_id, prompt, nodes, edges
        """
        nodes = [self._build_node(t, img_h, img_w, frame_bgr) for t in tracks]

        # Update motion history before computing motion relations
        for n in nodes:
            tid = n["track_id"]
            self._history.setdefault(tid, deque(maxlen=10))
            self._history[tid].append(
                (n["cx_norm"], n["cy_norm"], n["area_norm"])
            )

        # Add motion + heading attributes to each node
        for n in nodes:
            motion_attrs = self._motion_attrs(n["track_id"])
            n["motion"]       = motion_attrs["motion"]
            n["heading_vec"]  = motion_attrs["heading_vec"]
            n["heading_deg"]  = motion_attrs["heading_deg"]

        # Pairwise edges
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                rels = self._compute_relations(nodes[i], nodes[j], tracks[i], tracks[j])
                if rels:
                    edges.append({
                        "source": nodes[i]["track_id"],
                        "target": nodes[j]["track_id"],
                        "relations": rels,
                    })

        frame_graph = {
            "frame_id": frame_id,
            "prompt": self.text_prompt,
            "num_tracks": len(nodes),
            "nodes": nodes,
            "edges": edges,
        }
        self.frames.append(frame_graph)
        return frame_graph

    def save_jsonl(self, path: str):
        """Write all frames as newline-delimited JSON."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            for fg in self.frames:
                f.write(json.dumps(fg) + "\n")
        print(f"[SceneGraph] Saved {len(self.frames)} frames → {path}")

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate statistics across all processed frames."""
        if not self.frames:
            return {"total_frames": 0}
        total_nodes = sum(fg["num_tracks"] for fg in self.frames)
        total_edges = sum(len(fg["edges"]) for fg in self.frames)
        n = len(self.frames)
        return {
            "total_frames": n,
            "avg_nodes_per_frame": round(total_nodes / n, 2),
            "avg_edges_per_frame": round(total_edges / n, 2),
            "total_unique_tracks": len(self._history),
        }

    # ------------------------------------------------------------------
    # Node builder
    # ------------------------------------------------------------------

    def _build_node(
        self,
        track,
        img_h: int,
        img_w: int,
        frame_bgr: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        x, y, w, h = track.tlwh
        cx = x + w / 2.0
        cy = y + h / 2.0

        cx_norm   = cx / img_w
        cy_norm   = cy / img_h
        w_norm    = w  / img_w
        h_norm    = h  / img_h
        area_norm = (w * h) / (img_w * img_h)

        node: Dict[str, Any] = {
            "track_id":    int(track.track_id),
            "bbox_tlwh":   [round(float(v), 1) for v in (x, y, w, h)],
            "cx_norm":     round(cx_norm, 4),
            "cy_norm":     round(cy_norm, 4),
            "w_norm":      round(w_norm, 4),
            "h_norm":      round(h_norm, 4),
            "area_norm":   round(area_norm, 6),
            "confidence":  round(float(track.score), 3),
            "tracklet_len": int(getattr(track, "tracklet_len", 0)),
            "region":      self._region_label(cx_norm, cy_norm),
            "size":        self._size_label(area_norm),
            "has_embedding": (
                getattr(track, "embedding", None) is not None
            ),
        }

        if frame_bgr is not None:
            color, votes = _get_track_color(frame_bgr, x, y, w, h)
            node["color"]        = color
            node["color_votes"]  = votes

        return node

    # ------------------------------------------------------------------
    # Edge / relation computation
    # ------------------------------------------------------------------

    def _compute_relations(
        self, n1: dict, n2: dict, t1=None, t2=None
    ) -> List[str]:
        relations = []

        dx = n1["cx_norm"] - n2["cx_norm"]  # positive → n1 is to the RIGHT
        dy = n1["cy_norm"] - n2["cy_norm"]  # positive → n1 is BELOW

        # --- horizontal ---
        if dx < -self.SPATIAL_THRESH:
            relations.append("left-of")
        elif dx > self.SPATIAL_THRESH:
            relations.append("right-of")

        # --- vertical ---
        if dy < -self.SPATIAL_THRESH:
            relations.append("above")
        elif dy > self.SPATIAL_THRESH:
            relations.append("below")

        # --- proximity ---
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist <= self.NEAR_THRESH:
            relations.append("near")
        elif dist >= self.FAR_THRESH:
            relations.append("far")

        # --- size ---
        r = (n1["area_norm"] + 1e-9) / (n2["area_norm"] + 1e-9)
        if r > self.SIZE_RATIO:
            relations.append("larger-than")
        elif r < 1.0 / self.SIZE_RATIO:
            relations.append("smaller-than")

        # --- overlap ---
        iou = _iou_tlwh(n1["bbox_tlwh"], n2["bbox_tlwh"])
        if iou >= self.OVERLAP_THRESH:
            relations.append("overlapping")

        # --- visual similarity (CLIP embeddings) ---
        if t1 is not None and t2 is not None:
            e1 = getattr(t1, "embedding", None)
            e2 = getattr(t2, "embedding", None)
            if e1 is not None and e2 is not None:
                import torch
                sim = float(torch.dot(e1, e2).clamp(-1, 1))
                if sim >= self.CLIP_SIM_THRESH:
                    relations.append("visually-similar")

        return relations

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _region_label(cx_norm: float, cy_norm: float) -> str:
        h = "left" if cx_norm < 0.33 else ("right" if cx_norm > 0.67 else "center")
        v = "top"  if cy_norm < 0.33 else ("bottom" if cy_norm > 0.67 else "middle")
        return f"{v}-{h}"

    @staticmethod
    def _size_label(area_norm: float) -> str:
        if area_norm < 0.005:
            return "tiny"
        elif area_norm < 0.02:
            return "small"
        elif area_norm < 0.08:
            return "medium"
        return "large"

    def _motion_attrs(self, track_id: int) -> Dict[str, Any]:
        """Return motion label, heading vector and heading angle from position history.

        heading_vec: [dx, dy] unit vector in normalised image coords (right=+x, down=+y).
                     [0, 0] when stationary or insufficient history.
        heading_deg: angle in degrees; 0=right, 90=down, ±180=left, -90=up.
                     None when stationary or insufficient history.
        """
        hist = self._history.get(track_id)
        if hist is None or len(hist) < 3:
            return {"motion": "new", "heading_vec": [0.0, 0.0], "heading_deg": None}

        positions = list(hist)
        # Use full window for heading stability
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        # Area change over short window for approaching/receding
        da = positions[-1][2] - positions[-3][2]

        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.01:
            motion = "stationary"
            heading_vec = [0.0, 0.0]
            heading_deg = None
        else:
            inv = 1.0 / dist
            heading_vec = [round(dx * inv, 4), round(dy * inv, 4)]
            heading_deg = round(math.degrees(math.atan2(dy, dx)), 1)
            if da > 0.001:
                motion = "approaching"
            elif da < -0.001:
                motion = "receding"
            else:
                motion = "moving"

        return {"motion": motion, "heading_vec": heading_vec, "heading_deg": heading_deg}


# ---------------------------------------------------------------------------
# SceneGraphMissionFilter
# ---------------------------------------------------------------------------

# Color keywords recognised in prompts
_COLOR_KEYWORDS = ["red", "blue", "green", "yellow", "orange", "white", "black",
                   "gray", "grey", "dark", "light", "silver"]

# Spatial keywords → normalised region constraint
_SPATIAL_KEYWORDS = {
    "top":    ("cy_norm", "max", 0.40),   # cy_norm < 0.40
    "upper":  ("cy_norm", "max", 0.40),
    "bottom": ("cy_norm", "min", 0.60),
    "lower":  ("cy_norm", "min", 0.60),
    "left":   ("cx_norm", "max", 0.45),
    "right":  ("cx_norm", "min", 0.55),
    "center": ("cx_norm", "range", (0.25, 0.75)),
}


class SceneGraphMissionFilter:
    """
    Post-track filter driven by the scene graph.

    For each frame, call decide(frame_graph) → set of track_ids that satisfy
    the mission constraints parsed from the text prompt.

    Constraints currently supported (parsed automatically from the prompt):
      - color  : e.g. "red" → requires non-zero red votes in color_votes
      - region : e.g. "top" → requires cy_norm < 0.40

    Temporal accumulation: color votes are summed across a rolling window of
    HISTORY_LEN frames so that a single misclassified frame doesn't drop a track.

    Scoring:
      Each track gets a score in [0, 1] per constraint.  The final keep/drop
      decision uses a soft threshold (default 0.10) so borderline cases are
      kept rather than silently dropped.
    """

    HISTORY_LEN = 15        # frames of color vote history to accumulate
    COLOR_MIN_RATIO = 0.05  # fraction of accumulated votes that must be target color
    REGION_MARGIN = 0.08    # extra slack added to region bounds

    def __init__(self, text_prompt: str, hard_mode: bool = False,
                 score_thresh: float = 0.10):
        """
        Args:
            text_prompt: mission description, e.g. "red sedan at top".
            hard_mode:   if True, any failing constraint immediately rejects the
                         track.  If False (default), uses score_thresh.
            score_thresh: minimum combined score to keep a track (soft mode only).
        """
        self.text_prompt = text_prompt.lower()
        self.hard_mode = hard_mode
        self.score_thresh = score_thresh

        self.color_constraint: Optional[str] = self._parse_color()
        self.region_constraints: List[tuple] = self._parse_region()

        # track_id -> deque of color_votes dicts
        self._color_history: Dict[int, deque] = {}

        print(f"[MissionFilter] prompt='{text_prompt}' | "
              f"color={self.color_constraint} | "
              f"region={self.region_constraints} | "
              f"mode={'hard' if hard_mode else f'soft(thresh={score_thresh})'}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, frame_graph: Dict[str, Any]) -> set:
        """
        Return set of track_ids to keep from this frame's scene graph.

        Args:
            frame_graph: dict returned by SceneGraphBuilder.update()
        """
        kept = set()
        for node in frame_graph.get("nodes", []):
            tid = node["track_id"]
            self._accumulate_color(tid, node.get("color_votes", {}))
            score = self._score_node(node, tid)
            if self.hard_mode:
                if score >= 1.0:
                    kept.add(tid)
            else:
                if score >= self.score_thresh:
                    kept.add(tid)
        return kept

    def get_track_color_evidence(self, track_id: int) -> Dict[str, Any]:
        """Return accumulated color vote stats for a track (for debugging)."""
        hist = self._color_history.get(track_id)
        if not hist:
            return {}
        total: Dict[str, int] = {}
        for votes in hist:
            for color, count in votes.items():
                total[color] = total.get(color, 0) + count
        grand = sum(total.values()) or 1
        return {c: round(v / grand, 3) for c, v in sorted(total.items(),
                                                            key=lambda x: -x[1])}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_node(self, node: dict, track_id: int) -> float:
        scores = []

        # --- color constraint ---
        if self.color_constraint:
            scores.append(self._color_score(track_id))

        # --- region constraints ---
        for axis, op, bound in self.region_constraints:
            scores.append(self._region_score(node, axis, op, bound))

        if not scores:
            return 1.0  # no constraints → keep everything
        if self.hard_mode:
            return 1.0 if all(s >= 0.5 for s in scores) else 0.0
        return float(sum(scores) / len(scores))

    def _color_score(self, track_id: int) -> float:
        """Fraction of accumulated votes for the target color."""
        hist = self._color_history.get(track_id)
        if not hist:
            return 1.0  # no evidence yet → don't penalise
        total_target = 0
        total_all = 0
        for votes in hist:
            total_target += votes.get(self.color_constraint, 0)
            total_all += sum(votes.values())
        if total_all == 0:
            return 1.0
        ratio = total_target / total_all
        # Normalise: 0 at ratio=0, 1 at ratio >= COLOR_MIN_RATIO*4
        return min(1.0, ratio / max(self.COLOR_MIN_RATIO, 1e-6))

    @staticmethod
    def _region_score(node: dict, axis: str, op: str, bound) -> float:
        """1.0 if the node satisfies the spatial constraint, graded otherwise."""
        val = node.get(axis, 0.5)
        margin = SceneGraphMissionFilter.REGION_MARGIN
        if op == "max":
            # val should be < bound; full score at val=0, zero at val=bound+margin
            limit = bound + margin
            return max(0.0, min(1.0, (limit - val) / (limit + 1e-9)))
        elif op == "min":
            # val should be > bound; full score at val=1, zero at val=bound-margin
            limit = bound - margin
            return max(0.0, min(1.0, (val - limit) / (1.0 - limit + 1e-9)))
        elif op == "range":
            lo, hi = bound
            lo -= margin
            hi += margin
            if lo <= val <= hi:
                return 1.0
            dist = min(abs(val - lo), abs(val - hi))
            return max(0.0, 1.0 - dist / (margin + 1e-9))
        return 1.0

    # ------------------------------------------------------------------
    # Color history
    # ------------------------------------------------------------------

    def _accumulate_color(self, track_id: int, votes: dict):
        if track_id not in self._color_history:
            self._color_history[track_id] = deque(maxlen=self.HISTORY_LEN)
        self._color_history[track_id].append(dict(votes))

    # ------------------------------------------------------------------
    # Prompt parsing
    # ------------------------------------------------------------------

    def _parse_color(self) -> Optional[str]:
        for kw in _COLOR_KEYWORDS:
            if kw in self.text_prompt:
                # normalise grey → gray
                return "gray" if kw == "grey" else kw
        return None

    def _parse_region(self) -> List[tuple]:
        constraints = []
        for kw, spec in _SPATIAL_KEYWORDS.items():
            if kw in self.text_prompt:
                constraints.append(spec)
        return constraints
