"""
Fashion-aware clothing localizer.

Supports:
- garment retrieval
- upper/lower/full outfit retrieval
- unseen image inference
- rider jacket retrieval
- fashion-aware cropping
"""

from __future__ import annotations

from PIL import Image

try:

    from ultralytics import YOLO as _YOLO

    _HAS_ULTRALYTICS = True

except ImportError:

    _HAS_ULTRALYTICS = False


# =========================================================
# CUDA
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

    def __init__(
        self,
        weights: str = "weights/fashion_yolo.pt",
        conf_thresh: float = 0.20,
        iou_thresh: float = 0.45,
        device: str = "",
        padding_frac: float = 0.02,
    ):

        if not _HAS_ULTRALYTICS:

            raise ImportError(
                "ultralytics not installed"
            )

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.padding_frac = padding_frac

        print(
            f"[Fashion YOLO] Loading: {weights}"
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
            "[Fashion YOLO] Ready."
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
            }

        det = detections[0]

        return {

            "box":
                det["bbox"],

            "confidence":
                det["confidence"],

            "class_name":
                det["label"],

            "cropped":
                det["crop"],
        }

    # =====================================================
    # MULTI GARMENT DETECT
    # =====================================================

    def detect_all(
        self,
        image: Image.Image,
        max_regions: int = 10,
    ):

        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
        )[0]

        outputs = []

        W, H = image.size

        # =================================================
        # YOLO DETECTIONS
        # =================================================

        if (
            results.boxes is not None
            and
            len(results.boxes) > 0
        ):

            boxes = results.boxes

            for i, conf in enumerate(
                boxes.conf.tolist()
            ):

                cls_id = int(
                    boxes.cls[i].item()
                )

                label_name = str(
                    results.names[cls_id]
                ).lower()

                # =========================================
                # REMOVE NON-FASHION CLASSES
                # =========================================

                blocked = {

                    "person",
                    "human",
                    "face",
                }

                if label_name in blocked:

                    continue

                # =========================================
                # BBOX
                # =========================================

                x1, y1, x2, y2 = (

                    boxes.xyxy[i]
                    .cpu()
                    .numpy()
                    .astype(int)
                    .tolist()
                )

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(W, x2)
                y2 = min(H, y2)

                if x2 <= x1 or y2 <= y1:

                    continue

                bw = x2 - x1
                bh = y2 - y1

                # =========================================
                # TIGHTER CROPS
                # =========================================

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

                    "confidence": float(conf),

                    "label": label_name,

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
        # SEMANTIC REGIONS
        # =================================================

        upper_crop = image.crop((
            0,
            0,
            W,
            int(H * 0.55)
        ))

        lower_crop = image.crop((
            0,
            int(H * 0.45),
            W,
            H
        ))

        semantic_regions = [

            {
                "bbox": [
                    0,
                    0,
                    W,
                    H
                ],

                "confidence": 1.0,

                "label": "full_outfit",

                "crop": image,
            },

            {
                "bbox": [
                    0,
                    0,
                    W,
                    int(H * 0.55)
                ],

                "confidence": 1.0,

                "label": "upper_body",

                "crop": upper_crop,
            },

            {
                "bbox": [
                    0,
                    int(H * 0.45),
                    W,
                    H
                ],

                "confidence": 1.0,

                "label": "lower_body",

                "crop": lower_crop,
            },
        ]

        # =================================================
        # MERGE
        # =================================================

        final_outputs = semantic_regions + outputs

        return final_outputs[:max_regions]

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
