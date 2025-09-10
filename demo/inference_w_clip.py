import cv2
import time
import contextlib
import torch
import numpy as np
from PIL import Image
import clip
import torch.nn.functional as F
from torch.cuda.amp import autocast

from groundingdino.util.inference import load_model, predict
from tracker.tracker_w_clip import BYTETracker  # your custom version

# ===== Config (minimal) =====
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinB_cfg.py"
WEIGHTS_PATH = "weights/groundingdino_swinb_cogcoor.pth"
TEXT_PROMPT = "red car ."

BOX_THRESHOLD   = 0.42
TEXT_THRESHOLD  = 0.60
TRACK_THRESH    = 0.41
MATCH_THRESH    = 0.87
LAMBDA_WEIGHT   = 0.25
TEXT_SIM_THRESH = 0.25
TRACK_BUFFER_DEFAULT = 180

DEBUG = True
CLIP_PAD = 4  # pad crops a bit for more stable embeddings

# ===== Tiny latency helper =====
class Lat:
    def __init__(self): self.acc = {k:0.0 for k in ["decode","preproc","dino","post","clip","track","draw","frame"]}; self.n=0
    def add(self,k,dt): self.acc[k]+=float(dt)
    def rep(self):
        n=max(1,self.n); tot=self.acc["frame"]; fps=n/tot if tot>0 else 0
        ms=lambda k:1000*self.acc[k]/n
        print(f"[PERF] avg/frame: {1000*tot/n:.1f} ms ({fps:.2f} FPS) | "
              f"dino {ms('dino'):.1f} | clip {ms('clip'):.1f} | pre {ms('preproc'):.1f} | "
              f"dec {ms('decode'):.1f} | trk {ms('track'):.1f} | draw {ms('draw'):.1f}")

@contextlib.contextmanager
def tic(lat, key): t=time.perf_counter(); yield; lat.add(key, time.perf_counter()-t)

# ===== Preprocess (CUDA) -> [1,3,H,W] =====
def preprocess_cuda(frame_bgr, out_h, out_w, fp16, device):
    x = torch.from_numpy(frame_bgr).to(device, non_blocking=True)  # [H,W,3] u8
    x = x.permute(2,0,1).unsqueeze(0).contiguous(memory_format=torch.channels_last).float()  # [1,3,H,W]
    x = x.div_(255.0)
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    x = (x - mean) / std
    x = torch.nn.functional.interpolate(x, size=(out_h,out_w), mode='bilinear', align_corners=False)
    return x.half() if fp16 else x

# ===== NMS & box conversion =====
def _is_xyxy_pixels(boxes):
    if isinstance(boxes, np.ndarray): return boxes.size>0 and float(np.max(boxes))>1.5
    return boxes.numel()>0 and float(boxes.max().item())>1.5

def nms_xyxy(dets, iou=0.7):
    if dets.size==0: return []
    x1,y1,x2,y2,s = dets[:,0],dets[:,1],dets[:,2],dets[:,3],dets[:,4]
    a=(x2-x1+1)*(y2-y1+1); o=s.argsort()[::-1]; keep=[]
    while o.size>0:
        i=o[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[o[1:]]); yy1=np.maximum(y1[i],y1[o[1:]])
        xx2=np.minimum(x2[i],x2[o[1:]]); yy2=np.minimum(y2[i],y2[o[1:]])
        w=np.maximum(0.0,xx2-xx1+1); h=np.maximum(0.0,yy2-yy1+1)
        inter=w*h; iouv=inter/(a[i]+a[o[1:]]-inter+1e-6)
        inds=np.where(iouv<=iou)[0]; o=o[inds+1]
    return keep

def to_xyxy_dets(boxes, logits, W, H, min_side=8, max_area_frac=0.35, max_ar=4.0, nms_iou=0.7):
    if isinstance(boxes, torch.Tensor): boxes=boxes.detach().cpu().float().numpy()
    if isinstance(logits, torch.Tensor): logits=logits.detach().cpu().float().numpy()
    dets=[]
    def push(x1,y1,x2,y2,s):
        x1=max(0,min(float(x1),W-1)); y1=max(0,min(float(y1),H-1))
        x2=max(0,min(float(x2),W-1)); y2=max(0,min(float(y2),H-1))
        if x2<=x1 or y2<=y1: return
        w,h=(x2-x1),(y2-y1)
        if w<min_side or h<min_side: return
        if w*h > max_area_frac*(W*H): return
        ar=max(w/(h+1e-6), h/(w+1e-6))
        if ar>max_ar: return
        dets.append([x1,y1,x2,y2,float(s)])
    if _is_xyxy_pixels(boxes):
        for (x1,y1,x2,y2),s in zip(boxes,logits): push(x1,y1,x2,y2,s)
    else:
        for (cx,cy,w,h),s in zip(boxes,logits):
            if w<=0 or h<=0: continue
            x1=(cx-w/2)*W; y1=(cy-h/2)*H; x2=(cx+w/2)*W; y2=(cy+h/2)*H; push(x1,y1,x2,y2,s)
    if not dets: return np.zeros((0,5),np.float32)
    dets=np.asarray(dets,np.float32); keep=nms_xyxy(dets,nms_iou)
    return dets[keep] if keep else np.zeros((0,5),np.float32)

# ===== Draw =====
def draw_tracks(img, tracks, scale=1.0):
    for t in tracks:
        x,y,w,h=map(float,t.tlwh)
        if scale!=1.0: x,y,w,h = x*scale, y*scale, w*scale, h*scale
        x,y,w,h=map(int,(x,y,w,h))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,f"ID:{t.track_id}",(x,max(0,y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    return img

# ===== Helpers =====
def parse_classes(prompt): return [c.strip() for c in prompt.split('.') if c.strip()]

class TrackerArgs:
    def __init__(self, track_thresh=TRACK_THRESH, track_buffer=TRACK_BUFFER_DEFAULT,
                 match_thresh=MATCH_THRESH, lambda_weight=LAMBDA_WEIGHT, text_sim_thresh=TEXT_SIM_THRESH):
        self.track_thresh=float(track_thresh); self.track_buffer=int(track_buffer)
        self.match_thresh=float(match_thresh); self.lambda_weight=float(lambda_weight)
        self.text_sim_thresh=float(text_sim_thresh); self.aspect_ratio_thresh=10.0
        self.min_box_area=100; self.mot20=False

def compute_size(W,H,resize_long):
    if resize_long<=0: return W,H,1.0
    L=max(W,H)
    if resize_long>=L: return W,H,1.0
    s=resize_long/float(L); return int(round(W*s)), int(round(H*s)), s

# ===== Main =====
def main(video_path, output_path, fp16=False, resize_long=0, track_buffer=TRACK_BUFFER_DEFAULT):
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark=True
    lat=Lat()

    # Models
    model=load_model(CONFIG_PATH, WEIGHTS_PATH).to(device).eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device); clip_model.eval()

    # Text emb
    classes=parse_classes(TEXT_PROMPT) or ["object"]
    with torch.no_grad():
        te=clip_model.encode_text(clip.tokenize(classes).to(device))
        text_emb=F.normalize(te.float(),dim=-1).contiguous()

    cap=cv2.VideoCapture(video_path)
    W,H=int(cap.get(3)),int(cap.get(4)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    PW,PH,_=compute_size(W,H,resize_long); sx,sy=(W/float(PW), H/float(PH)) if (PW,PH)!=(W,H) else (1.0,1.0)
    out=cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))

    tracker=BYTETracker(TrackerArgs(track_buffer=track_buffer), frame_rate=fps)

    f=0
    while True:
        with tic(lat,"frame"):
            with tic(lat,"decode"):
                ok, frame = cap.read()
            if not ok: break
            f+=1

            # ==== preprocess → DINO (every frame)
            with tic(lat,"preproc"):
                img_t = preprocess_cuda(frame, PH, PW, fp16, device)
                if device=="cuda": torch.cuda.synchronize()
            with tic(lat,"dino"):
                with torch.no_grad(), autocast(enabled=fp16):
                    boxes, logits, _ = predict(model, img_t[0], TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD)
                if device=="cuda": torch.cuda.synchronize()
            with tic(lat,"post"):
                dets = to_xyxy_dets(boxes, logits, PW, PH)
                if DEBUG and f % 30 == 1: print(f"[Frame {f}] DINO dets:", len(dets))

            # ==== batch CLIP crops
            with tic(lat,"clip"):
                embeds=[]
                if len(dets)>0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    crops=[]
                    for x1,y1,x2,y2,_ in dets.tolist():
                        xi1,yi1=int(round(x1*sx))-CLIP_PAD, int(round(y1*sy))-CLIP_PAD
                        xi2,yi2=int(round(x2*sx))+CLIP_PAD, int(round(y2*sy))+CLIP_PAD
                        xi1=max(0,xi1); yi1=max(0,yi1); xi2=min(W,xi2); yi2=min(H,yi2)
                        c = rgb[yi1:yi2, xi1:xi2]
                        crops.append(None if c.size==0 or (yi2-yi1)<10 or (xi2-xi1)<10 else Image.fromarray(c))
                    batch = [clip_preprocess(c).unsqueeze(0) for c in crops if c is not None]
                    if batch:
                        batch = torch.cat(batch,0).to(device, non_blocking=True)
                        with torch.no_grad(): em = clip_model.encode_image(batch)
                        if device=="cuda": torch.cuda.synchronize()
                        em = F.normalize(em, dim=-1).float().cpu()
                        j=0
                        for c in crops:
                            embeds.append(None if c is None else em[j]); 
                            if c is not None: j+=1

            # ==== tracker & draw
            with tic(lat,"track"):
                tracks = tracker.update(
                    detections=dets,
                    detection_embeddings=embeds,
                    img_info=(PH, PW),
                    text_embedding=text_emb,
                    class_names=classes,
                )
            with tic(lat,"draw"):
                out.write(draw_tracks(frame.copy(), tracks, scale=sx))

        lat.n+=1
        if DEBUG and f%30==1: lat.rep()

    cap.release(); out.release()
    print("\n=== FINAL LATENCY ==="); lat.rep()
    if device=="cuda":
        try: print(f"[PERF] peak CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")
        except: pass

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--resize-long", type=int, default=0, help="Detector input long side (0=original)")
    p.add_argument("--track-buffer", type=int, default=TRACK_BUFFER_DEFAULT)
    a=p.parse_args()
    main(a.video, a.output, fp16=a.fp16, resize_long=a.resize_long, track_buffer=a.track_buffer)
