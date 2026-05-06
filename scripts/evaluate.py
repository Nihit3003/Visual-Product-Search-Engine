"""
Batch Evaluation Script — Ablation Study
=========================================
Runs all three conditions (A, B, C) and reports Recall@K, NDCG@K, mAP@K
for K ∈ {5, 10, 15} with mean ± std across seeds.

Usage (Kaggle):
    python scripts/evaluate.py \
        --dataset_root  /kaggle/input/deepfashion-inshop \
        --index_base    /kaggle/working/index \
        --ckpt_path     /kaggle/working/checkpoints/clip_finetuned_best.pt \
        --query_folder  /kaggle/input/deepfashion-inshop \
        --output_dir    /kaggle/working/results \
        --seeds         42 2024 1337 7

Or evaluate a single query folder (demo script):
    python scripts/evaluate.py \
        --query_folder /path/to/query/images \
        --index_dir    /path/to/saved/index \
        --single_run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset   import parse_eval_partition, parse_item_ids, parse_bboxes, \
    DeepFashionDataset, get_clip_transform
from src.model     import VisualSearchModel
from src.index     import HNSWIndex
from src.metrics   import evaluate, evaluate_multi_seed, MetricResults
from src.blip_module import ITMReranker
from src.localizer import YOLOLocalizer
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--index_base",   required=True,
                   help="Directory containing condition_A_alpha1.0/ etc.")
    p.add_argument("--ckpt_path",    default=None)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--seeds",        nargs="+", type=int, default=[42, 2024, 1337, 7])
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--embed_dim",    type=int, default=256)
    p.add_argument("--top_k",        type=int, default=50,
                   help="Retrieve top-K from ANN, then re-rank")
    p.add_argument("--use_itm",      action="store_true", default=True)
    p.add_argument("--yolo_weights", default="yolov8n.pt")
    p.add_argument("--alpha_B",      type=float, default=0.6,
                   help="Alpha for condition B")
    p.add_argument("--alpha_C",      type=float, default=0.6,
                   help="Alpha for condition C")
    p.add_argument("--query_folder", default=None,
                   help="If set, evaluate images from this folder only")
    p.add_argument("--single_run",   action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────
#  Encode query images
# ─────────────────────────────────────────────

@torch.no_grad()
def encode_queries(
    model       : VisualSearchModel,
    image_paths : list[str],
    img_root    : str,
    img_to_bbox : dict,
    localizer,
    device      : str,
    batch_size  : int = 64,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns (embeddings array of shape (N, D), list of item_ids).
    """
    transform   = get_clip_transform(224, augment=False)
    embs_list   = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="  Encoding queries"):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []
        for rel_path in batch_paths:
            full = str(Path(img_root) / rel_path)
            try:
                img = Image.open(full).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))

            # YOLO crop or GT bbox
            if localizer is not None:
                result = localizer.detect(img)
                img = result["cropped"]
            elif rel_path in img_to_bbox:
                img = img.crop(img_to_bbox[rel_path])

            batch_tensors.append(transform(img))

        tensors = torch.stack(batch_tensors).to(device)
        embs    = model.encode_image(tensors).cpu().numpy()
        embs_list.append(embs)

    return np.concatenate(embs_list, axis=0)


# ─────────────────────────────────────────────
#  Single-condition evaluation
# ─────────────────────────────────────────────

def run_condition(
    condition   : str,
    alpha       : float,
    index_dir   : str,
    model       : VisualSearchModel,
    query_paths : list[str],
    query_items : list[str],
    img_root    : str,
    img_to_bbox : dict,
    localizer,
    reranker    : "ITMReranker | None",
    device      : str,
    args,
) -> MetricResults:
    print(f"\n[Eval] ── Condition {condition} (alpha={alpha}) ──")

    # Load index
    index = HNSWIndex.load(index_dir)
    print(f"[Eval] Index loaded: {len(index):,} vectors")

    # Encode queries
    model.alpha = alpha if condition != "A" else 1.0
    q_embs = encode_queries(
        model, query_paths, img_root, img_to_bbox, localizer, device, args.batch_size
    )

    # Retrieve top-K candidates
    all_retrieved = []
    for q_emb in tqdm(q_embs, desc="  Searching index"):
        candidates = index.search(q_emb, top_k=args.top_k)

        # ITM re-ranking for conditions B and C
        if reranker is not None and condition in ("B", "C") and candidates:
            q_img_path = query_paths[len(all_retrieved)]
            try:
                q_img = Image.open(str(Path(img_root) / q_img_path)).convert("RGB")
            except Exception:
                q_img = Image.new("RGB", (224, 224))
            candidates = reranker.rerank(q_img, candidates)

        all_retrieved.append([c["item_id"] for c in candidates])

    # Compute metrics
    results = evaluate(
        query_ids=query_items,
        retrieved=all_retrieved,
        gallery_ids=[],
        item_to_imgs={},
        K_values=[5, 10, 15],
    )
    print(results)
    return results


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.dataset_root)
    splits      = parse_eval_partition(str(root / "list_eval_partition.txt"))
    img_to_item = parse_item_ids(str(root / "list_description_inshop.txt"))
    img_to_bbox = parse_bboxes(str(root / "list_bbox_inshop.txt"))

    query_paths = splits["query"]
    query_items = [img_to_item[p] for p in query_paths if p in img_to_item]
    query_paths = [p for p in query_paths if p in img_to_item]
    print(f"[Eval] Queries: {len(query_paths):,}")

    # ── Load models ──────────────────────────
    model_A = VisualSearchModel(alpha=1.0, embed_dim=args.embed_dim, unfreeze_last_n=0)
    model_A.eval().to(device)

    model_BC = VisualSearchModel(alpha=args.alpha_C, embed_dim=args.embed_dim, unfreeze_last_n=4)
    if args.ckpt_path:
        state = torch.load(args.ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model_BC.load_state_dict(state, strict=False)
    model_BC.eval().to(device)

    # Frozen BLIP-2 model for conditions B
    model_B = VisualSearchModel(alpha=args.alpha_B, embed_dim=args.embed_dim, unfreeze_last_n=0)
    model_B.eval().to(device)

    localizer = None
    try:
        localizer = YOLOLocalizer(weights=args.yolo_weights, device=device)
    except Exception as e:
        print(f"[Eval] YOLO unavailable: {e}")

    reranker = None
    if args.use_itm:
        try:
            reranker = ITMReranker(device=device)
        except Exception as e:
            print(f"[Eval] ITM reranker unavailable: {e}")

    # ── Run all conditions across seeds ──────
    all_results = {"A": [], "B": [], "C": []}

    for seed in args.seeds:
        print(f"\n[Eval] ════ Seed {seed} ════")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for cond, alpha, model in [
            ("A", 1.0,         model_A),
            ("B", args.alpha_B, model_B),
            ("C", args.alpha_C, model_BC),
        ]:
            idx_dir = str(Path(args.index_base) / f"condition_{cond}_alpha{alpha}")
            if not Path(idx_dir).exists():
                print(f"[Eval] Index not found for condition {cond}: {idx_dir} — skipping")
                continue

            res = run_condition(
                condition=cond, alpha=alpha,
                index_dir=idx_dir,
                model=model,
                query_paths=query_paths, query_items=query_items,
                img_root=str(root), img_to_bbox=img_to_bbox,
                localizer=localizer, reranker=reranker,
                device=device, args=args,
            )
            all_results[cond].append(res)

    # ── Aggregate across seeds ────────────────
    print("\n\n" + "═" * 60)
    print("  FINAL RESULTS (mean ± std across seeds)")
    print("═" * 60)

    final = {}
    for cond in ("A", "B", "C"):
        if all_results[cond]:
            agg = evaluate_multi_seed(all_results[cond])
            print(f"\nCondition {cond}:")
            print(agg)
            final[f"condition_{cond}"] = agg.to_dict()

    # ── Save JSON results ────────────────────
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[Eval] Results saved → {out_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
