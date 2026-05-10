"""
Final upgraded offline indexing pipeline.

Consistent with:
- ViT-L-14
- 768-d embeddings
- GT bbox support
- multimodal fusion
- upgraded HNSW
- hard-negative fine-tuning
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import (
    parse_eval_partition,
    parse_bboxes,
    get_clip_transform,
    infer_category,
    bbox_crop,
)

from src.model import VisualSearchModel

from src.blip_module import FashionCaptioner

from src.localizer import YOLOLocalizer

from src.index import HNSWIndex


# =========================================================
# ARGS
# =========================================================

def get_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset_root",
        required=True
    )

    p.add_argument(
        "--captions_json",
        default=None
    )

    p.add_argument(
        "--ckpt_path",
        default=None
    )

    p.add_argument(
        "--index_dir",
        required=True
    )

    p.add_argument(
        "--condition",
        choices=["A", "B", "C"],
        default="C"
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.7
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=48
    )

    p.add_argument(
        "--embed_dim",
        type=int,
        default=768
    )

    p.add_argument(
        "--use_gt_bbox",
        action="store_true",
        default=True
    )

    p.add_argument(
        "--yolo_weights",
        default="yolov8n.pt"
    )

    p.add_argument(
        "--unfreeze_last_n",
        type=int,
        default=6
    )

    return p.parse_args()


# =========================================================
# SAFE LOAD
# =========================================================

def open_image_safe(path):

    try:

        return Image.open(
            path
        ).convert("RGB")

    except Exception:

        return None


# =========================================================
# MODEL
# =========================================================

def load_model(
    args,
    device
):

    alpha = (
        1.0
        if args.condition == "A"
        else args.alpha
    )

    model = VisualSearchModel(
        alpha=alpha,
        unfreeze_last_n=args.unfreeze_last_n,
        embed_dim=args.embed_dim,
    )

    if (
        args.ckpt_path
        and
        args.condition == "C"
    ):

        ckpt = torch.load(
            args.ckpt_path,
            map_location=device
        )

        if "model_state_dict" in ckpt:

            ckpt = ckpt["model_state_dict"]

        model.load_state_dict(
            ckpt,
            strict=False
        )

        print(
            "[Indexing] Loaded checkpoint"
        )

    model.eval().to(device)

    return model


# =========================================================
# MULTI-CROP EMBEDDINGS
# =========================================================

@torch.no_grad()
def build_multi_crop_embeddings(
    images,
    model,
    transform,
    device,
):

    full_batch = []
    center_batch = []
    upper_batch = []

    for img in images:

        w, h = img.size

        full = img

        center = img.crop((
            int(0.15 * w),
            int(0.15 * h),
            int(0.85 * w),
            int(0.85 * h)
        ))

        upper = img.crop((
            0,
            0,
            w,
            int(0.7 * h)
        ))

        full_batch.append(
            transform(full)
        )

        center_batch.append(
            transform(center)
        )

        upper_batch.append(
            transform(upper)
        )

    full_batch = torch.stack(
        full_batch
    ).to(device)

    center_batch = torch.stack(
        center_batch
    ).to(device)

    upper_batch = torch.stack(
        upper_batch
    ).to(device)

    e1 = model.encode_image(
        full_batch
    )

    e2 = model.encode_image(
        center_batch
    )

    e3 = model.encode_image(
        upper_batch
    )

    emb = (
        0.5 * e1
        +
        0.3 * e2
        +
        0.2 * e3
    )

    emb = F.normalize(
        emb,
        dim=-1
    )

    return emb


# =========================================================
# MAIN
# =========================================================

def main():

    args = get_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_float32_matmul_precision(
        "high"
    )

    print(
        f"[Indexing] Device: {device}"
    )

    root = Path(args.dataset_root)

    # -----------------------------------------------------
    # captions
    # -----------------------------------------------------

    captions_map = {}

    if (
        args.captions_json is not None
        and
        Path(args.captions_json).exists()
    ):

        with open(args.captions_json) as f:

            captions_map = json.load(f)

        print(
            f"[Indexing] Loaded "
            f"{len(captions_map):,} captions"
        )

    # -----------------------------------------------------
    # parse dataset
    # -----------------------------------------------------

    splits, img_to_item = parse_eval_partition(
        root / "list_eval_partition.txt"
    )

    img_to_bbox = parse_bboxes(
        root / "list_bbox_inshop.txt"
    )

    gallery_paths = splits["gallery"]

    print(
        f"[Indexing] Gallery size: "
        f"{len(gallery_paths):,}"
    )

    # -----------------------------------------------------
    # model
    # -----------------------------------------------------

    model = load_model(
        args,
        device
    )

    total, trainable = model.param_count()

    print(
        f"[Indexing] "
        f"{trainable/1e6:.2f}M "
        f"trainable params"
    )

    # -----------------------------------------------------
    # captioner
    # -----------------------------------------------------

    captioner = None

    if args.condition in ("B", "C"):

        try:

            captioner = FashionCaptioner(
                device=device
            )

            print(
                "[Indexing] BLIP enabled"
            )

        except Exception as e:

            print(
                f"[Indexing] BLIP unavailable: {e}"
            )

    # -----------------------------------------------------
    # yolo
    # -----------------------------------------------------

    localizer = None

    if not args.use_gt_bbox:

        try:

            localizer = YOLOLocalizer(
                weights=args.yolo_weights,
                device=device,
            )

            print(
                "[Indexing] YOLO enabled"
            )

        except Exception as e:

            print(
                f"[Indexing] YOLO unavailable: {e}"
            )

    # -----------------------------------------------------
    # transform
    # -----------------------------------------------------

    transform = get_clip_transform(
        image_size=224,
        augment=False
    )

    # -----------------------------------------------------
    # index
    # -----------------------------------------------------

    index = HNSWIndex(
        dim=model.out_dim,
        M=64,
        ef_construction=400,
        ef_search=320,
    )

    batch_imgs = []
    batch_items = []
    batch_paths = []
    batch_metadata = []

    # =====================================================
    # FLUSH
    # =====================================================

    def flush_batch():

        if not batch_imgs:
            return

        # -------------------------------------------------
        # image embeddings
        # -------------------------------------------------

        img_embs = build_multi_crop_embeddings(
            batch_imgs,
            model,
            transform,
            device,
        )

        captions = []

        # -------------------------------------------------
        # captions
        # -------------------------------------------------

        for rel_path in batch_paths:

            cap = captions_map.get(
                rel_path,
                ""
            )

            if isinstance(cap, dict):

                cap = cap.get(
                    "caption",
                    ""
                )

            captions.append(
                cap
            )

        # -------------------------------------------------
        # fallback BLIP generation
        # -------------------------------------------------

        if (
            captioner is not None
            and
            any(len(c.strip()) == 0 for c in captions)
        ):

            try:

                generated = captioner.caption(
                    batch_imgs,
                    batch_size=2
                )

                for i in range(len(captions)):

                    if len(captions[i].strip()) == 0:

                        captions[i] = generated[i]

            except Exception as e:

                print(
                    f"[Indexing] Caption failure: {e}"
                )

        # -------------------------------------------------
        # multimodal fusion
        # -------------------------------------------------

        if (
            args.condition in ("B", "C")
            and
            any(len(c.strip()) > 0 for c in captions)
        ):

            try:

                text_embs = model.encode_text(
                    captions
                )

                text_embs = F.normalize(
                    text_embs,
                    dim=-1
                )

                embs = (
                    args.alpha * img_embs
                    +
                    (1.0 - args.alpha)
                    * text_embs
                )

                embs = F.normalize(
                    embs,
                    dim=-1
                )

            except Exception as e:

                print(
                    f"[Indexing] Fusion failed: {e}"
                )

                embs = img_embs

        else:

            embs = img_embs

        # -------------------------------------------------
        # add to index
        # -------------------------------------------------

        index.add(
            embeddings=embs.detach().cpu().numpy(),
            item_ids=batch_items[:],
            img_paths=batch_paths[:],
            captions=captions,
            metadata=batch_metadata[:],
        )

        batch_imgs.clear()
        batch_items.clear()
        batch_paths.clear()
        batch_metadata.clear()

    # =====================================================
    # LOOP
    # =====================================================

    t0 = time.time()

    for rel_path in tqdm(
        gallery_paths,
        desc="Indexing gallery"
    ):

        full_path = (
            root /
            "img" /
            rel_path
        )

        img = open_image_safe(
            full_path
        )

        if img is None:
            continue

        # -------------------------------------------------
        # localization
        # -------------------------------------------------

        conf = 1.0

        if (
            args.use_gt_bbox
            and
            rel_path in img_to_bbox
        ):

            crop = bbox_crop(
                img,
                img_to_bbox[rel_path]
            )

        elif localizer is not None:

            try:

                result = localizer.detect(
                    img
                )

                crop = result["cropped"]

                conf = float(
                    result.get(
                        "confidence",
                        0.5
                    )
                )

            except Exception:

                crop = img

        else:

            crop = img

        # -------------------------------------------------
        # metadata
        # -------------------------------------------------

        category = infer_category(
            rel_path
        )

        meta = {

            "category":
                category,

            "yolo_confidence":
                conf,

            "localization_quality":

                "high"
                if conf > 0.75 else

                "medium"
                if conf > 0.4 else

                "low",
        }

        batch_imgs.append(crop)

        batch_items.append(
            img_to_item.get(
                rel_path,
                "unknown"
            )
        )

        batch_paths.append(
            rel_path
        )

        batch_metadata.append(
            meta
        )

        # -------------------------------------------------
        # flush
        # -------------------------------------------------

        if len(batch_imgs) >= args.batch_size:

            flush_batch()

    flush_batch()

    elapsed = time.time() - t0

    print(
        f"\n[Indexing] Done! "
        f"{len(index):,} vectors "
        f"in {elapsed:.1f}s"
    )

    # =====================================================
    # SAVE
    # =====================================================

    save_path = (
        Path(args.index_dir)
        /
        f"condition_{args.condition}_alpha{args.alpha}"
    )

    index.save(
        str(save_path)
    )

    print(
        f"[Indexing] Saved → {save_path}"
    )

    cfg = vars(args)

    cfg["n_vectors"] = len(index)

    with open(
        save_path / "index_config.json",
        "w"
    ) as f:

        json.dump(
            cfg,
            f,
            indent=2
        )


if __name__ == "__main__":

    main()
