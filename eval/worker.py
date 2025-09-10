# worker.py
#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import importlib
import subprocess
from typing import Dict, Tuple, Optional, Iterable, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.cuda.amp import autocast
import clip

from groundingdino.util.inference import load_model, predict

# ----------------------------
# Defaults (override in __init__)
# ----------------------------
DEFAULT_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
DEFAULT_WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
DEFAULT_TEXT_PROMPT = "car. pedestrian."
DEFAULT_MIN_BOX_AREA = 10
DEFAULT_FRAME_RATE = 10

# Registry: tracker_name -> (module_path, class_name)
TRACKER_REGISTRY: Dict[str, Tuple[str, str]] = {
    "bytetrack": ("tracker.byte_tracker", "BYTETracker"),
    "clip": ("tracker.tracker_w_clip", "CLIPTracker"),
}

def _build_normalize_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

def _letterbox_tensor(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Pad to target (H, W) without distorting aspect ratio (centered)."""
    _, h, w = tensor.shape
    scale = min(size[0] / h, size[1] / w)
    resized = F.interpolate(
        tensor.unsqueeze(0),
        scale_factor=scale,
        mode="bilinear",
        align_corners=False
    )[0]
    pad_h = size[0] - resized.shape[1]
    pad_w = size[1] - resized.shape[2]
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return F.pad(resized, (left, right, top, bottom), value=0.0)

def _convert_dino_to_xyxy(
    boxes: Iterable[Iterable[float]],
    logits: Iterable[float],
    W: int,
    H: int
) -> np.ndarray:
    """
    DINO gives boxes in cx,cy,w,h (normalized 0..1). Convert to [x1,y1,x2,y2,score].
    """
    dets: List[List[float]] = []
    for box, logit in zip(boxes, logits):
        cx, cy, w, h = box
        if w <= 0 or h <= 0:
            continue
        score = float(logit)
        x1 = (cx - w / 2.0) * W
        y1 = (cy - h / 2.0) * H
        x2 = (cx + w / 2.0) * W
        y2 = (cy + h / 2.0) * H
        dets.append([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2), score])
    return np.array(dets, dtype=np.float32) if dets else np.empty((0, 5), dtype=np.float32)


def parse_kv_list(kv_list):
    """Parse repeatable --tracker_kv key=val into a typed dict."""
    out = {}
    for kv in kv_list or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            elif v.lower().startswith(("0x", "0b")):
                out[k] = int(v, 0)
            elif "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


class Worker:
    """
    Reusable GroundingDINO + Tracker runner.

    Key features:
      - Pluggable tracker via `tracker_type` ("bytetrack" or "clip", extendable).
      - Clean API usable from any eval script.
      - Saves MOTChallenge-style results per sequence.
    """

    def __init__(
        self,
        *,
        # DINO
        config_path: str = DEFAULT_CONFIG_PATH,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        text_prompt: str = DEFAULT_TEXT_PROMPT,
        box_thresh: float = 0.35,
        text_thresh: float = 0.25,
        use_fp16: bool = False,
        device: Optional[str] = None,

        # Tracker selection
        tracker_type: str = "bytetrack",  # "bytetrack" or "clip"
        tracker_kwargs: Optional[dict] = None,

        # Misc
        frame_rate: int = DEFAULT_FRAME_RATE,
        min_box_area: int = DEFAULT_MIN_BOX_AREA,
        verbose_first_n_frames: int = 5,
    ):
        self.text_prompt = text_prompt
        self.box_thresh = float(box_thresh)
        self.text_thresh = float(text_thresh)
        self.use_fp16 = bool(use_fp16)
        self.frame_rate = int(frame_rate)
        self.min_box_area = int(min_box_area)
        self.verbose_first_n_frames = int(verbose_first_n_frames)

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Model
        self.model = load_model(config_path, weights_path)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        # Preprocessing
        self._transform = _build_normalize_transform()

        # Tracker
        tracker_kwargs = dict(tracker_kwargs or {})
        tracker_namespace = argparse.Namespace(
            track_thresh=tracker_kwargs.pop("track_thresh", 0.5),
            track_buffer=tracker_kwargs.pop("track_buffer", 30),
            match_thresh=tracker_kwargs.pop("match_thresh", 0.8),
            aspect_ratio_thresh=tracker_kwargs.pop("aspect_ratio_thresh", 10.0),

            lambda_weight=tracker_kwargs.pop("lambda_weight", 0.25),
            text_sim_thresh=tracker_kwargs.pop("text_sim_thresh", 0.15),

            min_box_area=self.min_box_area,
            mot20=tracker_kwargs.pop("mot20", False),
            **tracker_kwargs,  # pass through tracker-specific extras
        )
        self.tracker = self._build_tracker(tracker_type, tracker_namespace, frame_rate=self.frame_rate)
        self.tracker_type = tracker_type

        # ---- CLIP init (only when using the CLIP-aware tracker) ----
        self.class_names: List[str] = [c.strip() for c in self.text_prompt.split(".") if c.strip()] or ["object"]
        self.text_embedding: Optional[torch.Tensor] = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_pad = int(tracker_kwargs.pop("clip_pad", 4))  # override via --tracker_kv clip_pad=6

        if self.tracker_type == "clip":
            # Load CLIP and precompute text embeddings
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            with torch.no_grad():
                tokens = clip.tokenize(self.class_names).to(self.device)
                te = self.clip_model.encode_text(tokens)
                self.text_embedding = F.normalize(te.float(), dim=-1).contiguous()

    @staticmethod
    def _build_tracker(tracker_type: str, tracker_args: argparse.Namespace, *, frame_rate: int):
        if tracker_type not in TRACKER_REGISTRY:
            raise ValueError(
                f"Unknown tracker_type '{tracker_type}'. "
                f"Available: {list(TRACKER_REGISTRY.keys())}"
            )
        module_path, class_name = TRACKER_REGISTRY[tracker_type]
        module = importlib.import_module(module_path)
        TrackerCls = getattr(module, class_name)
        return TrackerCls(tracker_args, frame_rate=frame_rate)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR np.array -> normalized tensor, letterboxed back to original size (H,W)."""
        h, w = frame_bgr.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = self._transform(img)
        tensor = _letterbox_tensor(tensor, size=(h, w))
        if str(self.device).startswith("cuda"):
            tensor = tensor.cuda(non_blocking=True)
        return tensor.half() if self.use_fp16 else tensor

    def predict_detections(self, tensor_image: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
        with torch.no_grad(), autocast(enabled=self.use_fp16):
            boxes, logits, _ = predict(
                model=self.model,
                image=tensor_image,
                caption=self.text_prompt,
                box_threshold=self.box_thresh,
                text_threshold=self.text_thresh,
            )
        return _convert_dino_to_xyxy(boxes, logits, orig_w, orig_h)

    def update_tracker(self, dets_xyxy: np.ndarray, orig_h: int, orig_w: int):
        """
        dets_xyxy: np.ndarray [N,5] as [x1,y1,x2,y2,score]
        returns: list of tracks from the underlying tracker
        """
        if dets_xyxy.size == 0:
            return self.tracker.update(np.empty((0, 5), dtype=np.float32), [orig_h, orig_w], [orig_h, orig_w])
        return self.tracker.update(dets_xyxy, [orig_h, orig_w], [orig_h, orig_w])

    def _compute_detection_embeddings(
        self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray, *, min_hw: int = 10
    ) -> List[Optional[torch.Tensor]]:
        """
        Crop detections from the original BGR frame and get CLIP image embeddings.
        Returns a list of CPU float tensors [D] or None for invalid crops.
        """
        if dets_xyxy.size == 0:
            return []
        assert self.clip_model is not None and self.clip_preprocess is not None, "CLIP not initialized."

        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        crops: List[Optional[Image.Image]] = []
        for (x1, y1, x2, y2, _s) in dets_xyxy.tolist():
            xi1, yi1 = int(round(x1)) - self.clip_pad, int(round(y1)) - self.clip_pad
            xi2, yi2 = int(round(x2)) + self.clip_pad, int(round(y2)) + self.clip_pad
            xi1 = max(0, xi1); yi1 = max(0, yi1)
            xi2 = min(W, xi2); yi2 = min(H, yi2)
            if xi2 <= xi1 or yi2 <= yi1 or (xi2 - xi1) < min_hw or (yi2 - yi1) < min_hw:
                crops.append(None)
            else:
                crops.append(Image.fromarray(rgb[yi1:yi2, xi1:xi2]))

        batch = [self.clip_preprocess(c).unsqueeze(0) for c in crops if c is not None]
        if not batch:
            return [None for _ in crops]

        batch_t = torch.cat(batch, 0).to(self.device, non_blocking=True)
        with torch.no_grad():
            em = self.clip_model.encode_image(batch_t)
        em = F.normalize(em, dim=-1).float().cpu()

        out: List[Optional[torch.Tensor]] = []
        j = 0
        for c in crops:
            out.append(None if c is None else em[j])
            if c is not None:
                j += 1
        return out

    def update_tracker_clip(self, dets_xyxy: np.ndarray, frame_bgr: np.ndarray, orig_h: int, orig_w: int):
        """
        CLIP-aware update: computes image embeddings for each detection and passes them,
        along with the precomputed text embeddings, to the CLIPTracker.
        """
        dets = dets_xyxy if dets_xyxy.size else np.empty((0, 5), dtype=np.float32)
        det_embs = self._compute_detection_embeddings(frame_bgr, dets)
        return self.tracker.update(
            detections=dets,
            detection_embeddings=det_embs,
            img_info=(orig_h, orig_w),
            text_embedding=self.text_embedding,
            class_names=self.class_names,
        )

    @staticmethod
    def _write_mot_line(fh, frame_id: int, track_id: int, x: float, y: float, w: float, h: float):
        fh.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    def process_sequence(
        self,
        *,
        seq: str,
        img_folder: str,
        out_path: str,
        sort_frames: bool = True,
    ):
        """
        Process a single sequence directory and write MOTChallenge-style results.

        Expects images in: os.path.join(img_folder, seq) with frame files named as integers.
        """
        seq_path = os.path.join(img_folder, seq)
        if not os.path.isdir(seq_path):
            raise FileNotFoundError(f"Sequence path does not exist: {seq_path}")

        frame_files = [f for f in os.listdir(seq_path) if os.path.isfile(os.path.join(seq_path, f))]
        if sort_frames:
            frame_files = sorted(frame_files, key=lambda n: int(os.path.splitext(n)[0]))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f_res:
            for idx, frame_name in enumerate(frame_files):
                frame_id = int(os.path.splitext(frame_name)[0])
                img = cv2.imread(os.path.join(seq_path, frame_name))
                if img is None:
                    continue
                orig_h, orig_w = img.shape[:2]

                tensor = self.preprocess_frame(img)

                if idx < self.verbose_first_n_frames:
                    _, proc_h, proc_w = tensor.shape
                    print(
                        f"[{seq}] Frame {frame_id}: "
                        f"Original {orig_h}x{orig_w} | Processed {proc_h}x{proc_w} | "
                        f"tracker={type(self.tracker).__name__}"
                    )

                dets = self.predict_detections(tensor, orig_h, orig_w)
                if self.tracker_type == "clip":
                    tracks = self.update_tracker_clip(dets, img, orig_h, orig_w)
                else:
                    tracks = self.update_tracker(dets, orig_h, orig_w)

                for t in tracks:
                    x, y, w, h = t.tlwh
                    if w * h > self.min_box_area:
                        self._write_mot_line(f_res, frame_id, t.track_id, float(x), float(y), float(w), float(h))

        print(f"[{seq}] Saved tracking results to: {out_path}")

    def process_many(
        self,
        *,
        seqs: Iterable[str],
        img_folder: str,
        res_folder: str,
        suffix: str = ".txt",
    ):
        os.makedirs(res_folder, exist_ok=True)
        for seq in seqs:
            out_path = os.path.join(res_folder, f"{seq}{suffix}")
            self.process_sequence(seq=seq, img_folder=img_folder, out_path=out_path)


# ----------------------------
# CLI (single/many + optional multi-GPU dispatch)
# ----------------------------
if __name__ == "__main__":
    import glob as _glob
    from datetime import datetime

    def list_sequences(img_root: str):
        return sorted([d for d in os.listdir(img_root)
                       if os.path.isdir(os.path.join(img_root, d))])

    def collect_sequences(args) -> List[str]:
        seqs = set()

        # explicit list on the CLI (can be multiple names)
        if args.seq:
            seqs.update(args.seq)

        # file with one seq per line
        if args.seq_file:
            with open(args.seq_file) as fh:
                for line in fh:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        seqs.add(s)

        # glob(s) under img_folder (e.g., "00*", "001*")
        if args.seq_glob:
            for pat in args.seq_glob:
                for p in _glob.glob(os.path.join(args.img_folder, pat)):
                    if os.path.isdir(p):
                        seqs.add(os.path.basename(p))

        # --all or nothing selected -> everything
        if args.all or not seqs:
            seqs.update(list_sequences(args.img_folder))

        return sorted(seqs)

    def resolve_single_out(seq: str, out_arg: Optional[str], out_dir: Optional[str], timestamp: bool) -> str:
        """
        If out_arg ends with .txt => treat as full file path.
        Else, write to (out_arg or out_dir)/(optional ts)/<seq>.txt
        """
        if out_arg and out_arg.lower().endswith(".txt"):
            os.makedirs(os.path.dirname(out_arg), exist_ok=True)
            return out_arg

        root = out_arg or out_dir or "outputs"
        if timestamp:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M")
            root = os.path.join(root, ts)
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, f"{seq}.txt")

    def dispatch_multi_gpu(seqs: List[str], args, tracker_kv: dict):
        """
        Spawn child worker.py processes, one per sequence, pinned to GPUs via CUDA_VISIBLE_DEVICES.
        """
        # parse devices list
        if args.devices:
            devices = [d.strip() for d in args.devices.split(",") if d.strip() != ""]
        else:
            # default to visible device 0
            devices = ["0"]
        if len(devices) == 0:
            devices = ["0"]

        jobs = max(1, int(args.jobs))
        procs: List[subprocess.Popen] = []

        # Where to put outputs
        root = args.out_dir or "outputs"
        if args.timestamp:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M")
            root = os.path.join(root, ts)
        os.makedirs(root, exist_ok=True)

        this_script = os.path.abspath(__file__)
        for i, seq in enumerate(seqs):
            gpu_id = devices[i % len(devices)]
            out_path = os.path.join(root, f"{seq}.txt")

            # build child cmd (single-seq mode)
            cmd = [
                sys.executable, "-u", this_script,
                "--seq", seq,
                "--img_folder", args.img_folder,
                "--out", out_path,
                "--tracker", args.tracker,
                "--box_thresh", str(args.box_thresh),
                "--text_thresh", str(args.text_thresh),
                "--track_thresh", str(args.track_thresh),
                "--match_thresh", str(args.match_thresh),
                "--track_buffer", str(args.track_buffer),
                "--text_prompt", args.text_prompt,
                "--config", args.config,
                "--weights", args.weights,
                "--min_box_area", str(args.min_box_area),
                "--frame_rate", str(args.frame_rate),
                "--child"  # mark as child to avoid redispatch
            ]
            if args.use_fp16:
                cmd.append("--use_fp16")
            for k, v in (tracker_kv or {}).items():
                cmd.extend(["--tracker_kv", f"{k}={v}"])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Silence Python warnings from child processes
            env["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::FutureWarning"
            # Quiet Transformers’ logger a bit
            env["TRANSFORMERS_VERBOSITY"] = "error"
            # Headless/quiet Matplotlib to avoid Axes3D/backends noise
            env["MPLBACKEND"] = "Agg"
            # Optional: reduce HF Hub chatter
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"

            p = subprocess.Popen(cmd, env=env)
            procs.append(p)

            # throttle concurrency
            if len(procs) >= jobs:
                procs[0].wait()
                procs = procs[1:]

        # drain
        for p in procs:
            p.wait()

    parser = argparse.ArgumentParser(description="Run GroundingDINO + pluggable tracker.")
    # Sequence selection (choose any combo; if none provided, --all is implied)
    parser.add_argument("--seq", nargs="*", help="Sequence name(s) under --img_folder (e.g., 0000 0001).")
    parser.add_argument("--seq_file", type=str, help="Text file with one sequence name per line.")
    parser.add_argument("--seq_glob", action="append",
                        help="Glob(s) to select sequences under --img_folder (repeatable), e.g. --seq_glob '00*'")
    parser.add_argument("--all", action="store_true", help="Process all sequences under --img_folder.")
    parser.add_argument("--img_folder", required=True, type=str, help="Root containing sequence subfolders")

    # Output controls
    parser.add_argument("--out", type=str, help="Output MOT file path (single sequence mode only). If not ending in .txt, treated as a directory.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory for per-sequence files")
    parser.add_argument("--timestamp", action="store_true", help="Append timestamp subfolder to outputs")

    # Model/tracker params
    parser.add_argument("--tracker", default="bytetrack", choices=list(TRACKER_REGISTRY.keys()))
    parser.add_argument("--box_thresh", type=float, default=0.35)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--text_prompt", type=str, default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--min_box_area", type=int, default=DEFAULT_MIN_BOX_AREA)
    parser.add_argument("--frame_rate", type=int, default=DEFAULT_FRAME_RATE)
    parser.add_argument("--tracker_kv", action="append", help="extra tracker args as key=val (repeatable)")

    # Built-in dispatcher (optional)
    parser.add_argument("--devices", type=str, help="Comma-separated GPU ids to use for dispatch, e.g. '0,1'. If set (and multiple seqs), worker will spawn one child per seq.")
    parser.add_argument("--jobs", type=int, default=1, help="Max concurrent child processes (<= #devices recommended).")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)  # internal flag

    args = parser.parse_args()

    tracker_kwargs = dict(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
    )
    tracker_kwargs.update(parse_kv_list(args.tracker_kv))

    # Child mode: run exactly one sequence with given --out
    if args.child:
        if not args.seq or len(args.seq) != 1 or not args.out:
            raise SystemExit("Child mode expects exactly one --seq and an --out file path.")
        worker = Worker(
            tracker_type=args.tracker,
            tracker_kwargs=tracker_kwargs,
            box_thresh=args.box_thresh,
            text_thresh=args.text_thresh,
            use_fp16=args.use_fp16,
            text_prompt=args.text_prompt,
            frame_rate=args.frame_rate,
            min_box_area=args.min_box_area,
            config_path=args.config,
            weights_path=args.weights,
        )
        out_path = args.out if args.out.lower().endswith(".txt") else os.path.join(args.out, f"{args.seq[0]}.txt")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        worker.process_sequence(seq=args.seq[0], img_folder=args.img_folder, out_path=out_path)
        raise SystemExit(0)

    # Parent / normal mode
    def sequences_or_all() -> List[str]:
        # build list of sequences (supports --all, --seq_glob, --seq_file)
        s = collect_sequences(args)
        return s

    seqs = sequences_or_all()

    # If devices provided AND more than one sequence -> dispatch across GPUs
    if args.devices and len(seqs) > 1:
        dispatch_multi_gpu(seqs, args, tracker_kwargs)
    else:
        # single or multi-seq on current process (no GPU dispatch)
        worker = Worker(
            tracker_type=args.tracker,
            tracker_kwargs=tracker_kwargs,
            box_thresh=args.box_thresh,
            text_thresh=args.text_thresh,
            use_fp16=args.use_fp16,
            text_prompt=args.text_prompt,
            frame_rate=args.frame_rate,
            min_box_area=args.min_box_area,
            config_path=args.config,
            weights_path=args.weights,
        )
        if len(seqs) == 1:
            out_path = resolve_single_out(seqs[0], args.out, args.out_dir, args.timestamp)
            worker.process_sequence(seq=seqs[0], img_folder=args.img_folder, out_path=out_path)
        else:
            root = args.out_dir
            if args.timestamp:
                ts = datetime.now().strftime("%Y-%m-%d_%H%M")
                root = os.path.join(root, ts)
            os.makedirs(root, exist_ok=True)
            for s in seqs:
                out_path = os.path.join(root, f"{s}.txt")
                worker.process_sequence(seq=s, img_folder=args.img_folder, out_path=out_path)
