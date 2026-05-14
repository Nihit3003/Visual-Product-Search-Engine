"""
Final DeepFashion Retrieval Evaluation

Consistent with:
- ViT-L-14
- 768-d embeddings
- GT bbox crops
- multi-crop indexing
- upgraded HNSW retrieval
"""

import argparse
import json
import sys
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

    p.add_argument(
        "--dataset_root",
        required=True,
    )

    p.add_argument(
        "--index_base",
        required=True,
    )

    p.add_argument(
        "--ckpt_path",
        default=None,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    p.add_argument(
        "--embed_dim",
        type=int,
        default=768,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=48,
    )

    p.add_argument(
        "--alpha_B",
        type=float,
        default=0.7,
    )

    p.add_argument(
        "--alpha_C",
        type=float,
        default=0.7,
    )

    p.add_argument(
        "--num_queries",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2, 536, 576, 584],
    )

    return p.parse_args()


# =========================================================
# MULTI-CROP QUERY EMBEDDINGS
# =========================================================

@torch.no_grad()
def encode_queries(
    model,
    query_paths,
    img_root,
    bbox_map,
    device,
    batch_size=48,
):

    model.eval()

    transform = get_clip_transform(
        224,
        augment=False
    )

    embs_list = []

    for i in tqdm(
        range(0, len(query_paths), batch_size),
        desc="Encoding queries",
    ):

        batch_paths = query_paths[
            i:i + batch_size
        ]

        full_batch = []
        center_batch = []
        upper_batch = []

        for rel_path in batch_paths:

            full_path = (
                Path(img_root)
                / "img"
                / rel_path
            )

            try:

                img = Image.open(
                    full_path
                ).convert("RGB")

            except Exception:

                img = Image.new(
                    "RGB",
                    (224, 224)
                )

            # -------------------------------------------------
            # GT bbox
            # -------------------------------------------------

            bbox = bbox_map.get(rel_path)

            if bbox is not None:

                img = bbox_crop(
                    img,
                    bbox
                )

            # -------------------------------------------------
            # multi-crop
            # -------------------------------------------------

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

        embs_list.append(
            emb.cpu().numpy()
        )

    return np.concatenate(
        embs_list,
        axis=0
    )


# =========================================================
# RUN CONDITION (FIXED)
# =========================================================

def run_condition(
    condition,
    alpha,
    model,
    index_dir,
    query_paths,
    query_items,
    gallery_items,
    img_root,
    bbox_map,
    device,
    args,
):

    print(
        f"\n[Eval] Condition "
        f"{condition} "
        f"(alpha={alpha})"
    )

    index = HNSWIndex.load(
        index_dir
    )

    print(
        f"[Eval] Loaded index "
        f"with {len(index):,} vectors"
    )

    model.alpha = (
        1.0
        if condition == "A"
        else alpha
    )

    q_embs = encode_queries(
        model=model,
        query_paths=query_paths,
        img_root=img_root,
        bbox_map=bbox_map,
        device=device,
        batch_size=args.batch_size,
    )

    retrieved = []

    # =====================================================
    # SEARCH
    # =====================================================

    for q_idx, q_emb in enumerate(
        tqdm(
            q_embs,
            desc="Searching"
        )
    ):

        query_path = query_paths[q_idx]

        query_item = query_items[q_idx]

        # -------------------------------------------------
        # retrieve MORE than needed
        # because duplicates/self-matches
        # will be filtered out
        # -------------------------------------------------

        candidates = index.search(
            q_emb,
            top_k=100,
            deduplicate_items=False,
        )

        unique_items = []

        seen_items = set()

        # -------------------------------------------------
        # CLEAN RANKING
        # -------------------------------------------------

        for c in candidates:

            item_id = c["item_id"]

            img_path = c["img_path"]

            # ---------------------------------------------
            # remove exact self-match
            # ---------------------------------------------

            if img_path == query_path:
                continue

            # ---------------------------------------------
            # keep only ONE result per item
            # ---------------------------------------------

            if item_id in seen_items:
                continue

            seen_items.add(item_id)

            unique_items.append(item_id)

            # ---------------------------------------------
            # stop once enough unique items
            # ---------------------------------------------

            if len(unique_items) == 15:
                break

        retrieved.append(unique_items)

    # =====================================================
    # EVALUATE
    # =====================================================

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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_float32_matmul_precision(
        "high"
    )

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    root = Path(
        args.dataset_root
    )

    splits, img_to_item = parse_eval_partition(
        root / "list_eval_partition.txt"
    )

    bbox_map = parse_bboxes(
        root / "list_bbox_inshop.txt"
    )

    query_paths = splits["query"][
        :args.num_queries
    ]

    query_paths = [

        p for p in query_paths

        if p in img_to_item

    ]

    query_items = [

        img_to_item[p]

        for p in query_paths

    ]

    gallery_items = [

        img_to_item[p]

        for p in splits["gallery"]

        if p in img_to_item
    ]

    print(
        f"[Eval] Queries: "
        f"{len(query_paths):,}"
    )

    # =====================================================
    # MODELS
    # =====================================================

    model_A = VisualSearchModel(
        clip_model_name="ViT-L-14",
        pretrained="openai",
        alpha=1.0,
        embed_dim=args.embed_dim,
        unfreeze_last_n=0,
    ).to(device)

    model_A.eval()

    model_B = VisualSearchModel(
        clip_model_name="ViT-L-14",
        pretrained="openai",
        alpha=args.alpha_B,
        embed_dim=args.embed_dim,
        unfreeze_last_n=0,
    ).to(device)

    model_B.eval()

    model_C = VisualSearchModel(
        clip_model_name="ViT-L-14",
        pretrained="openai",
        alpha=args.alpha_C,
        embed_dim=args.embed_dim,
        unfreeze_last_n=6,
    ).to(device)

    if args.ckpt_path:

        state = torch.load(
            args.ckpt_path,
            map_location=device
        )

        if "model_state_dict" in state:

            state = state[
                "model_state_dict"
            ]

        model_C.load_state_dict(
            state,
            strict=False
        )

    model_C.eval()

    # =====================================================
    # CONDITIONS
    # =====================================================

    configs = [

        (
            "A",
            1.0,
            model_A,
        ),

        (
            "B",
            args.alpha_B,
            model_B,
        ),

        (
            "C",
            args.alpha_C,
            model_C,
        ),
    ]

    all_results = {
        "A": [],
        "B": [],
        "C": [],
    }

    for seed in args.seeds:

        print(
            f"\n[Eval] "
            f"════ Seed {seed} ════"
        )

        torch.manual_seed(seed)

        np.random.seed(seed)

        for cond, alpha, model in configs:

            idx_dir = str(

                Path(args.index_base)

                / f"condition_{cond}_alpha{alpha}"

            )

            if not Path(idx_dir).exists():

                print(
                    f"[Eval] Missing index: "
                    f"{idx_dir}"
                )

                continue

            res = run_condition(

                condition=cond,

                alpha=alpha,

                model=model,

                index_dir=idx_dir,

                query_paths=query_paths,

                query_items=query_items,

                gallery_items=gallery_items,

                img_root=str(root),

                bbox_map=bbox_map,

                device=device,

                args=args,
            )

            all_results[cond].append(
                res
            )

    # =====================================================
    # AGGREGATE
    # =====================================================

    print(
        "\n"
        + "═" * 60
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "═" * 60
    )

    final = {}

    for cond in ["A", "B", "C"]:

        if not all_results[cond]:
            continue

        agg = evaluate_multi_seed(
            all_results[cond]
        )

        print(
            f"\nCondition {cond}"
        )

        print(agg)

        final[
            f"condition_{cond}"
        ] = agg.to_dict()

    # =====================================================
    # SAVE
    # =====================================================

    out_path = (
        out_dir
        / "ablation_results.json"
    )

    with open(
        out_path,
        "w"
    ) as f:

        json.dump(
            final,
            f,
            indent=2
        )

    print(
        f"\n[Eval] Saved → "
        f"{out_path}"
    )


if __name__ == "__main__":

    main()
