"""
DeepFashion Retrieval Evaluation
Matches the benchmark-style evaluation flow
used in the hard-negative mining notebook.
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
        default=64,
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
        default=[42, 83, 527, 588],
    )

    return p.parse_args()


# =========================================================
# QUERY ENCODING
# =========================================================

@torch.no_grad()
def encode_queries(
    model,
    query_paths,
    img_root,
    bbox_map,
    device,
    batch_size=64,
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

        batch_imgs = []

        for rel_path in batch_paths:

            full = (
                Path(img_root)
                / "img"
                / rel_path
            )

            try:

                img = Image.open(
                    full
                ).convert("RGB")

            except Exception:

                img = Image.new(
                    "RGB",
                    (224, 224)
                )

            # ---------------------------------------------
            # GT bbox crop ONLY
            # ---------------------------------------------

            bbox = bbox_map.get(rel_path)

            if bbox is not None:

                img = bbox_crop(
                    img,
                    bbox
                )

            batch_imgs.append(
                transform(img)
            )

        tensors = torch.stack(
            batch_imgs
        ).to(device)

        emb = model.encode_image(
            tensors
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
# RUN CONDITION
# =========================================================

def run_condition(
    condition,
    alpha,
    model,
    index_dir,
    query_paths,
    query_items,
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

    for q_emb in tqdm(
        q_embs,
        desc="Searching"
    ):

        candidates = index.search(
            q_emb,
            top_k=15,
            deduplicate_items=False,
        )

        retrieved.append([
            c["item_id"]
            for c in candidates
        ])

    results = evaluate(
        query_ids=query_items,
        retrieved=retrieved,
        gallery_ids=[],
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
        unfreeze_last_n=4,
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
