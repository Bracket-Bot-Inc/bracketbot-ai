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
    
    def __call__(self, source, conf=0.25, iou=0.45, imgsz=640, classes=None):
        orig_img = source.copy()
        speed = {}
        t1 = time.time()
        img, meta = self._preprocess(orig_img, imgsz)
        speed['preprocess'] = (time.time() - t1) * 1000
        t1 = time.time()
        outputs = self.rknn.inference(inputs=[img])
        speed['inference'] = (time.time() - t1) * 1000
        
        t1 = time.time()
        boxes = self._postprocess(outputs, conf, iou, orig_img.shape[:2], meta, classes)
        speed['postprocess'] = (time.time() - t1) * 1000
        
        return Results(boxes=boxes, names=self.names, orig_img=orig_img, speed=speed)
   
    # ────────────────────────── PRE-PROCESS ──────────────────────────

    def _preprocess(self, img, size=640):
        h0, w0 = img.shape[:2]
        r = min(size / h0, size / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        pad_w, pad_h = size - nw, size - nh
        img_resized = cv2.resize(img, (nw, nh), cv2.INTER_LINEAR)
        img_padded = cv2.copyMakeBorder(
            img_resized,
            pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT, value=(114,114,114),
        )
        meta = dict(ratio=r, pad=(pad_w//2, pad_h//2), imgsz=size)
        # RKNN expects NHWC uint8 input
        return img_padded[None].astype(np.uint8), meta



    # ───────────────────────── POST-PROCESS ─────────────────────────
    def _postprocess(self, outs, conf=0.25, iou=0.45, orig_shape=None, meta=None, classes=None):
        """
        Unified decoder + NMS for RKNN YOLO decoupled heads (YOLOv8/YOLO11 9-output).
        Returns (N,6): x1,y1,x2,y2,score,cls in pixel coords of the padded image, then
        unpads/unscales back to original using meta.
        """
        if not outs:
            return np.empty((0, 6), np.float32)

        def sigmoid(x): 
            return 1. / (1. + np.exp(-x))

        decoded_rows = []  # will collect [x1,y1,x2,y2,score,cls] rows
        imgsz = 640 if meta is None else int(meta.get('imgsz', 640))

        # If this is the 9-output decoupled head: [bbox, cls, obj] × 3 scales
        if len(outs) == 9:
            for i in range(0, 9, 3):
                bbox_pred = outs[i]      # (1, 64, H, W) -> DFL bins (16) for 4 sides
                cls_pred  = outs[i+1]    # (1, 80, H, W)
                obj_pred  = outs[i+2]    # (1, 1,  H, W)

                # shapes
                _, C, H, W = bbox_pred.shape
                assert C % 4 == 0, "bbox channels must be 4*dfl_len"
                dfl_len = C // 4
                stride  = max(1, imgsz // H)  # robust if imgsz!=640

                # softmax over dfl bins per side (axis=1 in (4,dfl,H,W))
                # reshape (1,64,H,W) -> (4, dfl, H, W)
                bb = bbox_pred[0].reshape(4, dfl_len, H, W)
                bb_max = bb.max(axis=1, keepdims=True)
                bb_exp = np.exp(bb - bb_max)
                bb_prob = bb_exp / bb_exp.sum(axis=1, keepdims=True)  # (4, dfl, H, W)
                bins = np.arange(dfl_len, dtype=np.float32).reshape(1, dfl_len, 1, 1)
                dist = (bb_prob * bins).sum(axis=1)  # (4, H, W) -> l,t,r,b in "bins"
                l, t, r, b = dist[0], dist[1], dist[2], dist[3]

                # centers in pixels
                xs = (np.arange(W, dtype=np.float32) + 0.5)
                ys = (np.arange(H, dtype=np.float32) + 0.5)
                grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W)
                cx = grid_x * stride
                cy = grid_y * stride

                # convert distances to pixels
                l *= stride; t *= stride; r *= stride; b *= stride
                x1 = cx - l
                y1 = cy - t
                x2 = cx + r
                y2 = cy + b

                # class/objectness
                obj = sigmoid(obj_pred[0, 0])      # (H, W)
                cls = sigmoid(cls_pred[0])         # (80, H, W)
                cls_best = cls.max(axis=0)         # (H, W)
                cls_idx  = cls.argmax(axis=0)      # (H, W)
                scores   = obj * cls_best          # (H, W)

                # threshold
                mask = scores > conf
                if not np.any(mask):
                    continue

                yy, xx = np.where(mask)
                pick_scores = scores[yy, xx]
                pick_cls    = cls_idx[yy, xx].astype(np.float32)
                pick_boxes  = np.stack([
                    x1[yy, xx], y1[yy, xx], x2[yy, xx], y2[yy, xx],
                    pick_scores, pick_cls
                ], axis=1)
                
                # Filter by classes if specified
                if classes is not None:
                    class_mask = np.isin(pick_boxes[:, 5], classes)
                    pick_boxes = pick_boxes[class_mask]
                
                if len(pick_boxes) > 0:
                    decoded_rows.append(pick_boxes)

        # If nothing decoded (unexpected format), return empty
        if not decoded_rows:
            return np.empty((0, 6), np.float32)

        boxes = np.concatenate(decoded_rows, axis=0).astype(np.float32)

        # NMS in padded-resized coords
        keep = self._nms(boxes[:, :4], boxes[:, 4], iou)
        boxes = boxes[keep]

        # Undo letterbox and resize back to original
        if meta is not None and orig_shape is not None:
            px, py = meta['pad']
            r      = meta['ratio']
            # subtract padding
            boxes[:, [0, 2]] -= px
            boxes[:, [1, 3]] -= py
            # scale back
            boxes[:, :4] /= r
            # clip to original image size
            H, W = orig_shape[:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H - 1)

        return boxes

    
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
