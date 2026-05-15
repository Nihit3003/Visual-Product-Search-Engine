"""
Fashionpedia-based clothing localizer.
Fine-grained garment detection for:
- jackets
- tshirts
- jeans
- dresses
- ties
- shoes
- handbags
etc.
"""

from __future__ import annotations

from PIL import Image

import torch

from transformers import pipeline


# =========================================================
# CUDA
# =========================================================

def _device():

    return 0 if torch.cuda.is_available() else -1


# =========================================================
# LOCALIZER
# =========================================================

class YOLOLocalizer:

    def __init__(
        self,
        weights: str = None,  # ADDED THIS to fix the TypeError from demo.py
        conf_thresh: float = 0.20,
        padding_frac: float = 0.02,
    ):

        self.conf_thresh = conf_thresh
        self.padding_frac = padding_frac

        print(
            "[Fashionpedia] Loading detector..."
        )

        self.detector = pipeline(
            "object-detection",
            model="valentinafeve/yolos-fashionpedia",
            device=_device(),
        )

        print(
            "[Fashionpedia] Ready."
        )

    # =====================================================
    # SINGLE DETECT
    # =====================================================

    def detect(
        self,
        image: Image.Image
    ):

        detections = self.detect_all(
            image,
            max_regions=1,
        )

        if len(detections) == 0:

            return {

                "box": None,
                "confidence": None,
                "class_name": None,
                "cropped": image,
            }

        det = detections[0]

        return {

            "box": det["bbox"],
            "confidence": det["confidence"],
            "class_name": det["label"],
            "cropped": det["crop"],
        }

    # =====================================================
    # MULTI DETECT
    # =====================================================

    def detect_all(
        self,
        image: Image.Image,
        max_regions: int = 10,
    ):

        outputs = []

        W, H = image.size

        detections = self.detector(image)

        for det in detections:

            score = float(det["score"])

            if score < self.conf_thresh:

                continue

            label = str(
                det["label"]
            ).lower()

            # =============================================
            # IGNORE NON-SEARCHABLE PARTS
            # =============================================

            IGNORE_LABELS = {

                # body parts
                "arm",
                "leg",
                "face",
                "hair",
                "skin",

                # tiny garment parts
                "sleeve",
                "neckline",
                "lapel",
                "pocket",
                "zipper",
                "button",
                "buckle",
                "bead",
                "bow",
                "fringe",
                "rivet",
                "ruffle",
                "sequin",
                "tassel",

                # duplicate tiny regions
                "collar",
                "cuff",

                # noisy detections
                "logo",
                "text",
            }

            if label in IGNORE_LABELS:

                continue

            box = det["box"]

            x1 = int(box["xmin"])
            y1 = int(box["ymin"])
            x2 = int(box["xmax"])
            y2 = int(box["ymax"])

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(W, x2)
            y2 = min(H, y2)

            if x2 <= x1 or y2 <= y1:

                continue

            bw = x2 - x1
            bh = y2 - y1

            # =============================================
            # TIGHTER CROPS
            # =============================================

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

                "confidence": score,

                "label": label,

                "crop": crop,
            })

        # =================================================
        # SORT
        # =================================================

        outputs = sorted(
            outputs,
            key=lambda x: x["confidence"],
            reverse=True
        )

        # =================================================
        # ALWAYS INCLUDE FULL OUTFIT
        # =================================================

        outputs.insert(0, {

            "bbox": [
                0,
                0,
                W,
                H
            ],

            "confidence": 1.0,

            "label": "full_outfit",

            "crop": image,
        })

        return outputs[:max_regions]

    # =====================================================
    # SIMPLE API
    # =====================================================

    def crop(
        self,
        image: Image.Image
    ):

        result = self.detect(
            image
        )

        return result["cropped"]

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
