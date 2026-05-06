"""
Offline Indexing Pipeline
=========================
Given the DeepFashion gallery split, this script:
  Step 1  : Detect + crop clothing items (YOLO) — or use GT bboxes as fallback
  Step 2  : Generate semantic captions (BLIP-2)   [optional; skip for condition A]
  Step 3  : Encode cropped images with fine-tuned CLIP
  Step 3b : Fuse image + text embeddings           [skip for condition A]
  Step 4  : Build HNSW index and persist to disk

Usage (Kaggle / terminal):
    python scripts/build_index.py \
        --dataset_root /kaggle/input/deepfashion-inshop \
        --ckpt_path    /kaggle/working/checkpoints/clip_finetuned.pt \
        --index_dir    /kaggle/working/index \
        --condition    C \
        --alpha        0.6 \
        --batch_size   64
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import parse_eval_partition, parse_bboxes
from src.model     import VisualSearchModel
from src.blip_module import FashionCaptioner
from src.localizer import YOLOLocalizer
from src.index     import HNSWIndex


# ─────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Build HNSW gallery index")
    p.add_argument("--dataset_root", required=True, help="Root of DeepFashion dataset")
    p.add_argument("--ckpt_path",    default=None,  help="Path to fine-tuned CLIP .pt")
    p.add_argument("--index_dir",    required=True, help="Where to save the FAISS index")
    p.add_argument("--condition",    choices=["A", "B", "C"], default="C",
                   help="A=vision-only, B=frozen CLIP+BLIP, C=finetuned CLIP+BLIP")
    p.add_argument("--alpha",        type=float, default=0.6,
                   help="Image-text fusion weight (ignored for condition A)")
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--embed_dim",    type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--use_gt_bbox",  action="store_true", default=True,
                   help="Use GT bboxes from list_bbox_inshop.txt as crop fallback")
    p.add_argument("--yolo_weights", default="yolov8n.pt")
    p.add_argument("--unfreeze_last_n", type=int, default=4)
    return p.parse_args()


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_model(args, device) -> VisualSearchModel:
    alpha = 1.0 if args.condition == "A" else args.alpha
    model = VisualSearchModel(
        alpha=alpha,
        unfreeze_last_n=args.unfreeze_last_n,
        embed_dim=args.embed_dim,
    )
    if args.ckpt_path and args.condition == "C":
        state = torch.load(args.ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[Indexing] Loaded fine-tuned CLIP from {args.ckpt_path}")
    model.eval().to(device)
    return model


def open_image_safe(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Indexing] Device: {device}")
    print(f"[Indexing] Condition: {args.condition}  |  alpha: {args.alpha}")

    root = Path(args.dataset_root)

    # ── Parse data files ─────────────────────
    splits, img_to_item = parse_eval_partition(str(root / "list_eval_partition.txt"))
    img_to_bbox = parse_bboxes(str(root / "list_bbox_inshop.txt"))

    gallery_paths = splits["gallery"]
    print(f"[Indexing] Gallery size: {len(gallery_paths):,}")

    # ── Load models ──────────────────────────
    model = load_model(args, device)
    total, trainable = model.param_count()
    print(f"[Indexing] Model params: {total/1e6:.1f}M total, {trainable/1e6:.2f}M trainable")

    localizer = None
    captioner = None

    if args.condition in ("B", "C"):
        captioner = FashionCaptioner(device=device)

    try:
        localizer = YOLOLocalizer(weights=args.yolo_weights, device=device)
    except Exception as e:
        print(f"[Indexing] YOLO not available ({e}), using GT bbox fallback.")

    # ── CLIP transform (no augmentation) ─────
    from src.dataset import get_clip_transform
    transform = get_clip_transform(image_size=224, augment=False)

    # ── Process gallery in batches ────────────
    index = HNSWIndex(dim=model.out_dim)
    batch_imgs, batch_items, batch_paths, batch_caps = [], [], [], []

    def flush_batch():
        if not batch_imgs:
            return
        # Encode images
        tensors = torch.stack([transform(img) for img in batch_imgs]).to(device)
        with torch.no_grad():
            img_embs = model.encode_image(tensors)

        if captioner is not None and args.condition in ("B", "C"):
            caps     = captioner.caption(batch_imgs)
            txt_embs = model.encode_text(caps)
            embs     = model.fuse(img_embs, txt_embs)
        else:
            caps = [""] * len(batch_imgs)
            embs = img_embs

        index.add(
            embeddings=embs.cpu().numpy(),
            item_ids=batch_items[:],
            img_paths=batch_paths[:],
            captions=caps,
        )
        batch_imgs.clear(); batch_items.clear()
        batch_paths.clear(); batch_caps.clear()

    t0 = time.time()
    for rel_path in tqdm(gallery_paths, desc="Indexing gallery"):
        full_path = str(root / rel_path)
        img = open_image_safe(full_path)
        if img is None:
            continue

        # YOLO crop or GT bbox fallback
        if localizer is not None:
            result = localizer.detect(img)
            img_crop = result["cropped"]
        elif args.use_gt_bbox and rel_path in img_to_bbox:
            img_crop = YOLOLocalizer.crop_from_gt(img, img_to_bbox[rel_path])
        else:
            img_crop = img

        batch_imgs.append(img_crop)
        batch_items.append(img_to_item.get(rel_path, "unknown"))
        batch_paths.append(rel_path)

        if len(batch_imgs) >= args.batch_size:
            flush_batch()

    flush_batch()   # remainder

    elapsed = time.time() - t0
    print(f"\n[Indexing] Done!  {len(index):,} vectors in {elapsed:.1f}s")

    # ── Save ────────────────────────────────
    save_path = Path(args.index_dir) / f"condition_{args.condition}_alpha{args.alpha}"
    index.save(str(save_path))
    print(f"[Indexing] Index saved → {save_path}")

    # Save config alongside the index
    cfg = vars(args)
    cfg["n_vectors"] = len(index)
    with open(save_path / "index_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
