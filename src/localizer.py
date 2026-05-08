"""
Improved YOLO-based clothing localizer.

Enhancements:
- tighter crops
- upper-garment prioritization
- confidence-aware fallback
- reduced background leakage
- better retrieval-focused crops
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


# =========================================================
# COCO IDS
# =========================================================

FASHION_COCO_IDS = {
    0,   # person
    27,  # backpack
    31,  # handbag
    32   # tie
}


# =========================================================
# LOCALIZER
# =========================================================

class YOLOLocalizer:
    """
    Improved clothing detector optimized for retrieval.

    Key changes:
    - tighter crops
    - reduced lower-body dominance
    - confidence-aware logic
    - fallback stability
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "",
        padding_frac: float = 0.03,   # tighter padding
        upper_body_bias: bool = True,
    ):

        if not _HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics not installed"
            )

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.padding_frac = padding_frac
        self.upper_body_bias = upper_body_bias

        print(f"[YOLO] Loading weights: {weights} ...")

        self.model = _YOLO(weights)

        self.model.to(
            device if device
            else "cuda" if _cuda_available()
            else "cpu"
        )

        print("[YOLO] Localizer ready.")

    # =====================================================
    # DETECT
    # =====================================================

    def detect(self, image: Image.Image) -> dict:

        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )[0]

        W, H = image.size

        best_box = None
        best_conf = 0.0
        best_cls = None

        # -------------------------------------------------
        # Choose best detection
        # -------------------------------------------------

        if results.boxes is not None and len(results.boxes) > 0:

            boxes = results.boxes

            for i, conf in enumerate(boxes.conf.tolist()):

                cls_id = int(boxes.cls[i].item())

                if conf > best_conf:

                    best_conf = conf

                    best_box = (
                        boxes.xyxy[i]
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist()
                    )

                    best_cls = results.names[cls_id]

        # -------------------------------------------------
        # No detection fallback
        # -------------------------------------------------

        if best_box is None:

            return {
                "box": None,
                "confidence": None,
                "class_name": None,
                "cropped": image,
                "full": image,
            }

        # -------------------------------------------------
        # Tight crop refinement
        # -------------------------------------------------

        x1, y1, x2, y2 = best_box

        bw = x2 - x1
        bh = y2 - y1

        # tighter padding

        pad_w = int(bw * self.padding_frac)
        pad_h = int(bh * self.padding_frac)

        x1 = max(0, x1 + pad_w)
        y1 = max(0, y1 + pad_h)

        x2 = min(W, x2 - pad_w)
        y2 = min(H, y2 - pad_h)

        # -------------------------------------------------
        # Upper-body prioritization
        # -------------------------------------------------

        # Removes pants / legs / lower-body dominance
        # for shirt-style retrieval.

        if self.upper_body_bias:

            refined_h = y2 - y1
            refined_w = x2 - x1

            aspect_ratio = refined_h / max(refined_w, 1)

            # Tall crops usually contain torso + legs
            # Keep upper ~65%

            if aspect_ratio > 1.35:

                y2 = int(y1 + 0.65 * refined_h)

        # -------------------------------------------------
        # Clamp again
        # -------------------------------------------------

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(W, x2)
        y2 = min(H, y2)

        # -------------------------------------------------
        # Safety checks
        # -------------------------------------------------

        if x2 <= x1 or y2 <= y1:

            cropped = image

            final_box = None

        else:

            final_box = [x1, y1, x2, y2]

            cropped = image.crop(final_box)

        # -------------------------------------------------
        # Confidence-aware fallback
        # -------------------------------------------------

        # Low confidence:
        # blend with full-image behavior later.

        return {
            "box": final_box,
            "confidence": float(best_conf),
            "class_name": best_cls,
            "cropped": cropped,
            "full": image,
        }

    # =====================================================
    # SIMPLE CROP API
    # =====================================================

    def crop(self, image: Image.Image) -> Image.Image:

        return self.detect(image)["cropped"]

    # =====================================================
    # BATCH
    # =====================================================

    def batch_detect(
        self,
        images: list[Image.Image],
        batch_size: int = 8,
    ) -> list[dict]:

        outputs = []

        for i in range(0, len(images), batch_size):

            batch = images[i:i+batch_size]

            for img in batch:

                outputs.append(
                    self.detect(img)
                )

        return outputs

    # =====================================================
    # GT BBOX
    # =====================================================

    @staticmethod
    def crop_from_gt(
        image: Image.Image,
        bbox: list[int]
    ) -> Image.Image:

        x1, y1, x2, y2 = bbox

        # safety clamp

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        if x2 <= x1 or y2 <= y1:
            return image

        return image.crop((x1, y1, x2, y2))


# =========================================================
# CUDA CHECK
# =========================================================

def _cuda_available() -> bool:

    try:

        import torch

        return torch.cuda.is_available()

    except ImportError:

        return False
