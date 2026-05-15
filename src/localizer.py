"""
High-quality YOLO clothing localizer.

Optimized for:
- retrieval accuracy
- ViT-L embeddings
- upper-garment focus
- reduced background leakage
- region-aware retrieval
- multi-item inference
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
    32,  # tie
}


# =========================================================
# CUDA CHECK
# =========================================================

def _cuda_available():

    try:

        import torch

        return torch.cuda.is_available()

    except Exception:

        return False


# =========================================================
# LOCALIZER
# =========================================================

class YOLOLocalizer:

    """
    Retrieval-oriented clothing localizer.

    Key improvements:
    - tighter retrieval crops
    - upper-body prioritization
    - multi-region outputs
    - confidence-aware fallbacks
    - reduced lower-body dominance
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_thresh: float = 0.22,
        iou_thresh: float = 0.45,
        device: str = "",
        padding_frac: float = 0.025,
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

        print(
            f"[YOLO] Loading: {weights}"
        )

        self.model = _YOLO(weights)

        self.model.to(

            device

            if device

            else "cuda"

            if _cuda_available()

            else "cpu"
        )

        print(
            "[YOLO] Ready."
        )

    # =====================================================
    # SINGLE DETECT
    # =====================================================

    def detect(
        self,
        image: Image.Image
    ) -> dict:

        detections = self.detect_all(
            image,
            max_regions=1,
        )

        if len(detections) == 0:

            return {

                "box":
                    None,

                "confidence":
                    None,

                "class_name":
                    None,

                "cropped":
                    image,

                "full":
                    image,

                "upper":
                    image,

                "lower":
                    image,
            }

        det = detections[0]

        crop = det["crop"]

        x1, y1, x2, y2 = det["bbox"]

        h_box = y2 - y1

        upper_crop = image.crop((
            x1,
            y1,
            x2,
            int(y1 + 0.58 * h_box)
        ))

        lower_crop = image.crop((
            x1,
            int(y1 + 0.42 * h_box),
            x2,
            y2
        ))

        return {

            "box":
                det["bbox"],

            "confidence":
                det["confidence"],

            "class_name":
                det["label"],

            "cropped":
                crop,

            "full":
                crop,

            "upper":
                upper_crop,

            "lower":
                lower_crop,
        }

    # =====================================================
    # MULTI DETECT
    # =====================================================

    def detect_all(
        self,
        image: Image.Image,
        max_regions: int = 6,
    ):

        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )[0]

        W, H = image.size

        outputs = []

        if (
            results.boxes is None
            or
            len(results.boxes) == 0
        ):

            return outputs

        boxes = results.boxes

        for i, conf in enumerate(
            boxes.conf.tolist()
        ):

            cls_id = int(
                boxes.cls[i].item()
            )

            xyxy = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )

            x1, y1, x2, y2 = xyxy

            # -------------------------------------------------
            # clamp
            # -------------------------------------------------

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(W, x2)
            y2 = min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            bw = x2 - x1
            bh = y2 - y1

            # -------------------------------------------------
            # padding reduction
            # -------------------------------------------------

            pad_w = int(
                bw * self.padding_frac
            )

            pad_h = int(
                bh * self.padding_frac
            )

            x1 = max(0, x1 + pad_w)
            y1 = max(0, y1 + pad_h)

            x2 = min(W, x2 - pad_w)
            y2 = min(H, y2 - pad_h)

            # -------------------------------------------------
            # upper-body prioritization
            # -------------------------------------------------

            if self.upper_body_bias:

                refined_h = y2 - y1
                refined_w = x2 - x1

                aspect_ratio = (
                    refined_h /
                    max(refined_w, 1)
                )

                if aspect_ratio > 1.25:

                    y2 = int(
                        y1 + 0.68 * refined_h
                    )

            crop = image.crop((
                x1,
                y1,
                x2,
                y2
            ))

            outputs.append({

                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "confidence": float(conf),

                "label": results.names[
                    cls_id
                ],

                "crop": crop,
            })

        # -------------------------------------------------
        # sort by confidence
        # -------------------------------------------------

        outputs = sorted(
            outputs,
            key=lambda x: x["confidence"],
            reverse=True
        )

        return outputs[:max_regions]

    # =====================================================
    # SIMPLE API
    # =====================================================

    def crop(
        self,
        image: Image.Image
    ):

        return self.detect(
            image
        )["cropped"]

    # =====================================================
    # BATCH
    # =====================================================

    def batch_detect(
        self,
        images,
        batch_size=8,
    ):

        outputs = []

        for i in range(
            0,
            len(images),
            batch_size
        ):

            batch = images[
                i : i + batch_size
            ]

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
        bbox
    ):

        x1, y1, x2, y2 = bbox

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        if x2 <= x1 or y2 <= y1:

            return image

        return image.crop((
            x1,
            y1,
            x2,
            y2
        ))
