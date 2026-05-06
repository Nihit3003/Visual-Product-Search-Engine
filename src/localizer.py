"""
YOLO-based clothing product localizer.

Wraps any Ultralytics YOLO model (v8/v9/v10/v11) to:
  1. Detect the primary clothing bounding box in an image.
  2. Return the cropped region for downstream CLIP / BLIP processing.
  3. Fall back to full image if no detection is confident enough.

YOLO weights remain FROZEN throughout the project.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO as _YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


# ─────────────────────────────────────────────
#  Clothing categories in COCO (used by YOLOv8n default weights)
# COCO class indices relevant to fashion:
#   27 = backpack, 31 = handbag, 32 = tie, 73 = umbrella (skip)
# We use a broad set and simply pick the highest-confidence detection.
# For fashion-specific YOLO, any custom clothing-trained weights also work.
# ─────────────────────────────────────────────

FASHION_COCO_IDS = {0, 27, 31, 32}   # person, backpack, handbag, tie


class YOLOLocalizer:
    """
    Detects the primary clothing item in an image and returns the crop.

    Args:
        weights      : path to YOLO .pt weights  OR  a model name like 'yolov8n.pt'
        conf_thresh  : minimum confidence to accept a detection
        iou_thresh   : NMS IoU threshold
        device       : 'cuda' | 'cpu' | '' (auto)
        padding_frac : fractional padding added around the detected box
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "",
        padding_frac: float = 0.05,
    ):
        if not _HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        self.conf_thresh  = conf_thresh
        self.iou_thresh   = iou_thresh
        self.padding_frac = padding_frac

        print(f"[YOLO] Loading weights: {weights} …")
        self.model = _YOLO(weights)
        self.model.to(device if device else "cuda" if _cuda_available() else "cpu")
        print("[YOLO] Localizer ready.")

    # ── Core detection ───────────────────────

    def detect(self, image: Image.Image) -> dict:
        """
        Run inference on a single PIL image.

        Returns:
            {
                'box'        : [x1, y1, x2, y2] (int pixels) or None,
                'confidence' : float or None,
                'class_name' : str or None,
                'cropped'    : PIL.Image (crop or full image),
                'full'       : PIL.Image (original),
            }
        """
        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )[0]

        W, H = image.size
        best_box  = None
        best_conf = 0.0
        best_cls  = None

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes
            # Prefer person / clothing classes; otherwise take highest-conf detection
            for i, conf in enumerate(boxes.conf.tolist()):
                cls_id = int(boxes.cls[i].item())
                # Accept all detections; fashion items will generally rank first
                if conf > best_conf:
                    best_conf = conf
                    best_box  = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    best_cls  = results.names[cls_id]

        # Pad the box and clamp to image bounds
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            pad_w = int((x2 - x1) * self.padding_frac)
            pad_h = int((y2 - y1) * self.padding_frac)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(W, x2 + pad_w)
            y2 = min(H, y2 + pad_h)
            best_box = [x1, y1, x2, y2]
            cropped  = image.crop(best_box)
        else:
            cropped = image   # fallback: full image

        return {
            "box"       : best_box,
            "confidence": best_conf if best_box else None,
            "class_name": best_cls,
            "cropped"   : cropped,
            "full"      : image,
        }

    def crop(self, image: Image.Image) -> Image.Image:
        """Convenience: return only the cropped PIL image."""
        return self.detect(image)["cropped"]

    # ── Batch processing ─────────────────────

    def batch_detect(
        self,
        images: list[Image.Image],
        batch_size: int = 8,
    ) -> list[dict]:
        """
        Run detection on a list of images in mini-batches.
        Returns a list of detect() result dicts.
        """
        results_all = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            for img in batch:
                results_all.append(self.detect(img))
        return results_all

    # ── GT-bbox fallback ─────────────────────

    @staticmethod
    def crop_from_gt(image: Image.Image, bbox: list[int]) -> Image.Image:
        """Use ground-truth bbox (from list_bbox_inshop.txt) instead of YOLO."""
        return image.crop(bbox)


# ─────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
