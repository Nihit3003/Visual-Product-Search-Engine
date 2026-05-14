"""
Final DeepFashion Retrieval Evaluation - Optimized
- Focuses on GT-bbox alignment
- Implemented Asymmetric Querying (Vision Query -> Fused Gallery)
- Deduplication by Item ID
"""

import argparse
import json
import sys
import os
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
    bbox_crop,
)
from src.model import VisualSearchModel
from src.index import HNSWIndex
from src.metrics import (
    evaluate,
    evaluate_multi_seed,
)

# =========================================================
# ARGS
# =========================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--index_base", required=True)
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--alpha_B", type=float, default=0.7)
    p.add_argument("--alpha_C", type=float, default=0.7)
    p.add_argument("--num_queries", type=int, default=1000)
    p.add_argument("--seeds", nargs="+", type=int, default=[2, 536, 576, 584])
    return p.parse_args()


# =========================================================
# CLEAN QUERY ENCODING
# =========================================================

@torch.no_grad()
def encode_queries(model, query_paths, img_root, bbox_map, device, batch_size=64):
    """
    CLEAN QUERY ENCODING: Uses single GT-bbox crop. 
    Pretrained models (A & B) perform significantly better with clean signals.
    """
    model.eval()
    transform = get_clip_transform(224, augment=False)
    embs_list = []

    for i in tqdm(range(0, len(query_paths), batch_size), desc="Encoding queries"):
        batch_paths = query_paths[i:i + batch_size]
        imgs_batch = []

        for rel_path in batch_paths:
            full_path = Path(img_root) / "img" / rel_path
            try:
                img = Image.open(full_path).convert("RGB")
                bbox = bbox_map.get(rel_path)
                if bbox is not None:
                    img = bbox_crop(img, bbox)
                imgs_batch.append(transform(img))
            except Exception:
                # Fallback for missing images
                imgs_batch.append(torch.zeros(3, 224, 224))

        imgs_tensor = torch.stack(imgs_batch).to(device)
        
        # Extract pure vision features
        emb = model.encode_image(imgs_tensor)
        emb = F.normalize(emb, dim=-1)
        embs_list.append(emb.cpu().numpy())

    return np.concatenate(embs_list, axis=0)


# =========================================================
# RUN CONDITION
# =========================================================

def run_condition(condition, alpha, model, index_dir, query_paths, query_items, gallery_items, img_root, bbox_map, device, args):
    print(f"\n[Eval] Condition {condition} (alpha={alpha})")

    index = HNSWIndex.load(index_dir)
    print(f"[Eval] Loaded index with {len(index):,} vectors")

    # ASYMMETRIC FIX: Force query to be vision-only (alpha=1.0) 
    # even if the gallery used text fusion (Condition B or C).
    original_alpha = model.alpha
    model.alpha = 1.0 

    q_embs = encode_queries(
        model=model,
        query_paths=query_paths,
        img_root=img_root,
        bbox_map=bbox_map,
        device=device,
        batch_size=args.batch_size,
    )
    
    # Restore model alpha
    model.alpha = original_alpha

    retrieved = []

    for q_idx, q_emb in enumerate(tqdm(q_embs, desc="Searching")):
        query_path = query_paths[q_idx]
        
        # Retrieve top 100 to ensure we have enough after deduplication
        candidates = index.search(q_emb, top_k=100)

        unique_items = []
        seen_items = set()

        for c in candidates:
            item_id = c["item_id"]
            img_path = c["img_path"]

            # 1. Remove exact self-match
            if img_path == query_path:
                continue

            # 2. Top-K Diversity: Keep only ONE result per item_id
            if item_id in seen_items:
                continue

            seen_items.add(item_id)
            unique_items.append(item_id)

            # 3. Stop once we hit K=15
            if len(unique_items) == 15:
                break

        retrieved.append(unique_items)

    # Evaluate using team's standard metrics
    results = evaluate(
        query_ids=query_items,
        retrieved=retrieved,
        gallery_ids=gallery_items,
        item_to_imgs={},
        K_values=[5, 10, 15],
    )

    print(results)
    return results


# =========================================================
# MAIN
# =========================================================

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.dataset_root)
    splits, img_to_item = parse_eval_partition(root / "list_eval_partition.txt")
    bbox_map = parse_bboxes(root / "list_bbox_inshop.txt")

    # Filter queries to requested count
    query_paths = [p for p in splits["query"][:args.num_queries] if p in img_to_item]
    query_items = [img_to_item[p] for p in query_paths]
    gallery_items = [img_to_item[p] for p in splits["gallery"] if p in img_to_item]

    print(f"[Eval] Valid Queries: {len(query_paths):,}")

    # Initialize Models for Ablation
    model_A = VisualSearchModel(
        clip_model_name="ViT-L-14", pretrained="openai", alpha=1.0, embed_dim=args.embed_dim, unfreeze_last_n=0
    ).to(device).eval()

    model_B = VisualSearchModel(
        clip_model_name="ViT-L-14", pretrained="openai", alpha=args.alpha_B, embed_dim=args.embed_dim, unfreeze_last_n=0
    ).to(device).eval()

    model_C = VisualSearchModel(
        clip_model_name="ViT-L-14", pretrained="openai", alpha=args.alpha_C, embed_dim=args.embed_dim, unfreeze_last_n=6
    ).to(device).eval()

    if args.ckpt_path:
        state = torch.load(args.ckpt_path, map_location=device)
        ckpt = state["model_state_dict"] if "model_state_dict" in state else state
        model_C.load_state_dict(ckpt, strict=False)
    
    model_C.eval()

    configs = [
        ("A", 1.0, model_A),
        ("B", args.alpha_B, model_B),
        ("C", args.alpha_C, model_C),
    ]

    all_results = {"A": [], "B": [], "C": []}

    for seed in args.seeds:
        print(f"\n[Eval] ════ Seed {seed} ════")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for cond, alpha, model in configs:
            idx_dir = str(Path(args.index_base) / f"condition_{cond}_alpha{alpha}")
            
            if not Path(idx_dir).exists():
                print(f"[Eval] Missing index: {idx_dir}")
                continue

            res = run_condition(
                condition=cond, alpha=alpha, model=model, index_dir=idx_dir,
                query_paths=query_paths, query_items=query_items, gallery_items=gallery_items,
                img_root=str(root), bbox_map=bbox_map, device=device, args=args,
            )
            all_results[cond].append(res)

    # Aggregate Final Results
    final = {}
    print("\n" + "═" * 60 + "\nFINAL SUMMARY\n" + "═" * 60)

    for cond in ["A", "B", "C"]:
        if not all_results[cond]: continue
        agg = evaluate_multi_seed(all_results[cond])
        print(f"\nCondition {cond}\n{agg}")
        final[f"condition_{cond}"] = agg.to_dict()

    out_path = out_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[Eval] Saved → {out_path}")

if __name__ == "__main__":
    main()
