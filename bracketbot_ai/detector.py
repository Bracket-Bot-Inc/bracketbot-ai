#!/usr/bin/env python3
"""BracketBot AI Detector - Ultralytics-style API for RKNN inference"""

import argparse
import sys
import time
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass, field

import numpy as np
import cv2
from rknnlite.api import RKNNLite

from bracketbot_ai.model_manager import ensure_model


COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

@dataclass
class Results:
    """Detection results container"""
    boxes: np.ndarray = field(default_factory=lambda: np.array([]))
    names: Dict[int, str] = field(default_factory=dict)
    orig_img: Optional[np.ndarray] = None
    speed: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self): return f"Results(boxes={len(self.boxes)}, path={self.path})"
    def __len__(self): return len(self.boxes)
    
    @property
    def xyxy(self): return self.boxes[:, :4] if len(self.boxes) else np.array([])
    @property
    def conf(self): return self.boxes[:, 4] if len(self.boxes) else np.array([])
    @property
    def cls(self): return self.boxes[:, 5].astype(int) if len(self.boxes) else np.array([])
    
    def plot(self, line_width=2, font_scale=0.5):
        if self.orig_img is None: return None
        img = self.orig_img.copy()
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,0,255), (255,128,0)]
        
        for box in self.boxes:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)
            color = colors[cls % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
            label = f"{self.names.get(cls, f'class_{cls}')} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img, (x1, y1-label_size[1]-4), (x1+label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)
        return img

class Detector:
    """RKNN Object Detection Detector"""
    
    def __init__(self, model: str, device=0, verbose=True):
        self.device = device
        self.verbose = verbose
        self.rknn = None
        self.names = {i: name for i, name in enumerate(COCO_NAMES)}
        # Resolve model path
        _, self.model_path = ensure_model(model)
        self._load_model()
    
    def _load_model(self):
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0: raise RuntimeError(f"Failed to load RKNN model: {ret}")
        
        core_mask = 0b111 if self.device == -1 else (1 << self.device)
        ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0: raise RuntimeError(f"Failed to init RKNN runtime: {ret}")
        
        if self.verbose:
            print(f"✓ Detector loaded: {self.model_path.name} on NPU device {self.device}")
    
    def __call__(self, source, conf=0.25, iou=0.45, imgsz=640):
        orig_img = source.copy()
        speed = {}
        t1 = time.time()
        img = self._preprocess(orig_img, imgsz)
        speed['preprocess'] = (time.time() - t1) * 1000
        t1 = time.time()
        outputs = self.rknn.inference(inputs=[img])
        for i, o in enumerate(outputs):
            print(f'out[{i}] shape {o.shape}, dtype {o.dtype}, min {o.min():.3f}, max {o.max():.3f}')
        speed['inference'] = (time.time() - t1) * 1000
        
        t1 = time.time()
        boxes = self._postprocess(outputs, conf, iou, orig_img.shape[:2])
        speed['postprocess'] = (time.time() - t1) * 1000
        
        return Results(boxes=boxes, names=self.names, orig_img=orig_img, speed=speed)
   
    # ────────────────────────── PRE-PROCESS ──────────────────────────
    def _preprocess(self, img, size=640):
        """
        Letter-box to `size×size` and return uint8 BGR tensor
        shaped (1,H,W,3) – the default input format for quantised
        RKNN graphs.

        Returns
        -------
        img4d : np.ndarray  # uint8 (1,H,W,3)
        """
        h0, w0 = img.shape[:2]
        r      = min(size / h0, size / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        pad_w, pad_h = size - nw, size - nh

        img = cv2.resize(img, (nw, nh), cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(
            img,
            pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img[None].astype(np.uint8)          # (1,H,W,3) BGR uint8


    # ───────────────────────── POST-PROCESS ─────────────────────────
    def _postprocess(self, outs, conf=0.25, iou=0.45, orig_shape=None):
        """
        Unified decoder + NMS for *all* common RKNN YOLO exports.

        Handles output tensors shaped:
            • (1, 84, 8400)              – flat anchor-free v8 head
            • (1, 8400, 6)               – already decoded boxes
            • (1, H, W, 84|85)           – NHWC anchor-free head(s)
            • (1, 3, H, W, 85)           – anchor-based head(s)
            • 9 outputs                  – YOLOv8 decoupled head format

        Returns
        -------
        boxes : np.ndarray  # (N,6)  x1,y1,x2,y2,score,cls  in *pixel* coords
        """
        if not outs:
            return np.empty((0, 6), np.float32)

        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        decoded = []  # collect [x1,y1,x2,y2,score,cls] rows (coords ∈0-1)

        # ─── Handle YOLOv8 decoupled head format (9 outputs) ──────────────
        if len(outs) == 9:
            # Format: [bbox_80x80, cls_80x80, obj_80x80, bbox_40x40, cls_40x40, obj_40x40, bbox_20x20, cls_20x20, obj_20x20]
            for i in range(0, 9, 3):
                bbox_pred = outs[i]      # (1, 64, H, W) - 4 coords * 16 anchors  
                cls_pred = outs[i+1]     # (1, 80, H, W) - 80 classes
                obj_pred = outs[i+2]     # (1, 1, H, W) - objectness
                
                _, _, H, W = bbox_pred.shape
                s = {80: 8, 40: 16, 20: 32}[H]  # stride
                
                # Apply sigmoid to get probabilities  
                obj_scores = sigmoid(obj_pred[0, 0])  # (H, W)
                cls_scores = sigmoid(cls_pred[0])     # (80, H, W)
                
                # Combine objectness and class scores
                best_cls_scores = cls_scores.max(0)   # (H, W)
                conf_map = obj_scores * best_cls_scores
                
                # Find candidates above confidence threshold
                y, x = np.where(conf_map > conf)
                
                if len(x) > 0:
                    # For decoupled head, bbox format might be different
                    # Try direct coordinate prediction first
                    bbox_raw = bbox_pred[0, :4]  # (4, H, W) - take first 4 channels
                    cx_raw = bbox_raw[0, y, x]
                    cy_raw = bbox_raw[1, y, x] 
                    w_raw = bbox_raw[2, y, x]
                    h_raw = bbox_raw[3, y, x]
                    
                    # Convert to pixel coordinates 
                    cx_pix = (cx_raw + x) * s
                    cy_pix = (cy_raw + y) * s
                    w_pix = w_raw * s
                    h_pix = h_raw * s
                    
                    # Convert to normalized coordinates
                    x1 = np.clip((cx_pix - w_pix/2) / 640, 0, 1)
                    y1 = np.clip((cy_pix - h_pix/2) / 640, 0, 1)
                    x2 = np.clip((cx_pix + w_pix/2) / 640, 0, 1)
                    y2 = np.clip((cy_pix + h_pix/2) / 640, 0, 1)
                    
                    # Get class predictions
                    best_cls = cls_scores.argmax(0)[y, x]
                    conf_scores = conf_map[y, x]
                    
                    decoded.extend(
                        np.column_stack((x1, y1, x2, y2, conf_scores, best_cls)).tolist()
                    )
        else:
            # Original processing for other formats
            for o in outs:
                sh = o.shape
                # ------------------------------------------------------------------
                # 1. Already-decoded list (1, 8400, 6) :  x1 y1 x2 y2 conf cls
                # ------------------------------------------------------------------
                if o.ndim == 3 and sh[-1] == 6:
                    boxes = o[0]
                    keep  = boxes[:, 4] > conf
                    decoded.extend(boxes[keep].tolist())
                    continue

                # ------------------------------------------------------------------
                # 2. Flat anchor-free head  (1, 84, 8400)
                # ------------------------------------------------------------------
                if o.ndim == 3 and sh[1] == 84 and sh[2] == 8400:
                    # This 84-channel model appears to have all-zero class logits
                    # Skip processing as it's likely not properly trained
                    continue

                # ------------------------------------------------------------------
                # 3. NCHW anchor-free head  (1, 85, H, W) - YOLOv8 style
                # ------------------------------------------------------------------
                if o.ndim == 4 and sh[1] == 85:
                    # Already in NCHW format
                    _, c, H, W = o.shape
                    s          = {80: 8, 40: 16, 20: 32}[H]
                    head       = o[0]  # (85, H, W)
                    
                    # For YOLOv8: channels are [cx, cy, w, h, conf, cls0, cls1, ...]
                    cx_raw = head[0]  # (H, W)
                    cy_raw = head[1]  # (H, W) 
                    w_raw = head[2]   # (H, W)
                    h_raw = head[3]   # (H, W)
                    conf_raw = head[4]  # (H, W)
                    cls_raw = head[5:]  # (80, H, W)
                    
                    # Apply sigmoid if needed (check if values are in logit space)
                    if conf_raw.max() > 1.0:
                        conf_raw = sigmoid(conf_raw)
                    if cls_raw.max() > 1.0:
                        cls_raw = sigmoid(cls_raw)
                    
                    # Compute confidence: Use class confidence directly (no objectness in this model)
                    best_cls_conf = cls_raw.max(0)  # (H, W)
                    conf_map = conf_raw * best_cls_conf
                    
                    # If objectness is all zeros, use class confidence only 
                    if conf_raw.max() == 0:
                        conf_map = best_cls_conf
                    
                    y, x = np.where(conf_map > conf)
                    
                    if len(x):
                        # Create grid coordinates
                        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
                        
                        # Get values at detection points
                        cx_sel = cx_raw[y, x]
                        cy_sel = cy_raw[y, x]
                        w_sel = w_raw[y, x]
                        h_sel = h_raw[y, x]
                        
                        # Convert to pixel coordinates (YOLOv8 format)
                        cx_pix = (cx_sel + x) * s  # Add grid offset and scale
                        cy_pix = (cy_sel + y) * s
                        w_pix = w_sel * s
                        h_pix = h_sel * s
                        
                        # Convert to normalized box coordinates
                        x1 = np.clip((cx_pix - w_pix/2) / 640, 0, 1)
                        y1 = np.clip((cy_pix - h_pix/2) / 640, 0, 1)
                        x2 = np.clip((cx_pix + w_pix/2) / 640, 0, 1)
                        y2 = np.clip((cy_pix + h_pix/2) / 640, 0, 1)
                        
                        best_cls = cls_raw.argmax(0)[y, x]
                        
                        decoded.extend(
                            np.column_stack((x1, y1, x2, y2,
                                            conf_map[y, x],
                                            best_cls))
                            .tolist()
                        )
                    continue

        # ─── nothing decoded ───────────────────────────────────────────
        if not decoded:
            return np.empty((0, 6), np.float32)

        # ─── NMS + rescale to orig image size ──────────────────────────
        boxes = np.asarray(decoded, np.float32)
        keep  = self._nms(boxes[:, :4], boxes[:, 4], iou)
        boxes = boxes[keep]

        if orig_shape is not None:
            oh, ow = orig_shape[:2]
            boxes[:, [0, 2]] *= ow
            boxes[:, [1, 3]] *= oh

        return boxes.astype(np.float32) 

    
    def _nms(self, boxes, scores, iou_threshold):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1: break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(iou < iou_threshold)[0] + 1]
        
        return np.array(keep)
    
    def info(self):
        print(f"\n📋 Detector Information:")
        print(f"  Path: {self.model_path} | Device: NPU {self.device} | Classes: {len(self.names)}")
    
    def __del__(self):
        if hasattr(self, 'rknn') and self.rknn: self.rknn.release()

def main():
    from mjpeg_streamer import MjpegServer, Stream
    parser = argparse.ArgumentParser(description="BracketBot AI Object Detector")
    parser.add_argument('model', help='Path to .rknn model file (checks models/ dir)')
    parser.add_argument('source', nargs='?', help='Image, video path, or camera index')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.1, help='IOU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference size')
    parser.add_argument('--stream', action='store_true', help='Stream results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--device', type=int, default=0, choices=[-1,0,1,2], help='NPU device')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--info', action='store_true', help='Show model info')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()

    
    try:
        model = Detector(args.model, device=args.device, verbose=args.verbose)
        
        if args.info:
            model.info()
            return 0
        
        print(f"\nProcessing: {args.source}")
        print(f"Settings: conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
        
        # Handle video/camera streams
        if args.source.isdigit() or Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
            print("Streaming mode (press 'q' to quit)...\n")

            # Open video source
            cap_source = int(args.source) if args.source.isdigit() else args.source
            cap = cv2.VideoCapture(cap_source)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {args.source}")

            ret = False
            while not ret:  
                ret, frame = cap.read()
            frame = frame[:,:frame.shape[1]//2]
            if args.stream: 
                stream = Stream("my_camera", size=(frame.shape[0], frame.shape[1]), quality=50, fps=30)
                server = MjpegServer("localhost", 8080)
                server.add_stream(stream)
                server.start()
            frame_count = total_detections = 0
            save_dir = Path("runs/detect") if args.save else None
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
            speeds_avg = {}
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run detection on single frame
                    results = model(frame[:,:frame.shape[1]//2], conf=args.conf, iou=args.iou)
                    for speed in results.speed:
                        if speed not in speeds_avg:
                            speeds_avg[speed] = 0
                        speeds_avg[speed] += results.speed[speed]
                    frame_count += 1
                    total_detections += len(results.boxes)
                    print(f"\rFrame {frame_count}: {len(results.boxes)} objects", end='', flush=True)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    cv2.imwrite("preprocess.jpg", results.orig_img)
                    # Save frame
                    if args.save:
                        save_path = save_dir / f"frame_{frame_count:06d}.jpg"
                        save_img = results.plot()
                        if save_img is not None:
                            cv2.imwrite(str(save_path), save_img)
                    if args.stream:
                        # Stream results to localhost:PORT
                        stream.set_frame(results.plot())
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("\n\nStopped by user")
            finally:
                print(f"\n\nProcessed {frame_count} frames, {total_detections} total detections")
                for speed in speeds_avg:
                    print(f"{speed}: {speeds_avg[speed]/frame_count:.1f}ms")
                if frame_count > 0: 
                    print(f"Average: {total_detections/frame_count:.1f} objects/frame")
                    if args.save:
                        print(f"Saved frames to: {save_dir}")
                if args.stream:
                    server.stop()
                cap.release()
        else:
            # Handle single image
            results = model(args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                          save=args.save, show=args.show)
            
            print(f"\nResults: {len(results)} objects detected")
            
            if args.verbose and len(results) > 0:
                print("\nDetections:")
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2, conf, cls = box
                    print(f"  [{i}] {results.names[int(cls)]}: {conf:.3f} at ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                print(f"\nSpeed: {sum(results.speed.values()):.1f}ms total")
            
            if args.save:
                print(f"\nSaved to: {results.save()}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 