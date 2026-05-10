"""
Hard-Negative CLIP Fine-Tuning Script
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import hnswlib
import numpy as np
import torch
import torch.nn.functional as F

from torch.cuda.amp import (
    GradScaler,
    autocast
)

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import (
    parse_eval_partition,
    parse_bboxes,
    get_clip_transform,
    HardNegTripletDataset,
)

from src.model import (
    VisualSearchModel,
    SupConLoss,
    TripletLoss,
)

from src.metrics import evaluate


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
        "--output_dir",
        required=True
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-5
    )

    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2
    )

    p.add_argument(
        "--embed_dim",
        type=int,
        default=256
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--unfreeze_last_n",
        type=int,
        default=4
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=0.07
    )

    p.add_argument(
        "--triplet_margin",
        type=float,
        default=0.3
    )

    p.add_argument(
        "--triplet_weight",
        type=float,
        default=0.5
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    return p.parse_args()


# =========================================================
# SEED
# =========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


# =========================================================
# EMBED ROWS
# =========================================================

@torch.no_grad()
def embed_rows(
    rows,
    model,
    transform,
    img_root,
    bbox_map,
    device,
    batch_size=128,
):

    model.eval()

    all_embs = []
    all_ids = []
    all_names = []

    for i in tqdm(
        range(0, len(rows), batch_size),
        desc="Embedding",
        leave=False
    ):

        batch = rows[i:i + batch_size]

        crops = []
        ids = []
        names = []

        for r in batch:

            path = img_root / r["image_name"]

            if not path.exists():
                continue

            try:

                from PIL import Image

                img = Image.open(
                    path
                ).convert("RGB")

                bbox = bbox_map.get(
                    r["image_name"]
                )

                if bbox is not None:

                    from src.dataset import bbox_crop

                    img = bbox_crop(
                        img,
                        bbox
                    )

                crops.append(
                    transform(img)
                )

                ids.append(
                    r["item_id"]
                )

                names.append(
                    r["image_name"]
                )

            except Exception:
                continue

        if not crops:
            continue

        t = torch.stack(crops).to(device)

        emb = model.encode_image(t)

        emb = emb.cpu().numpy()

        all_embs.append(emb)

        all_ids.extend(ids)

        all_names.extend(names)

    return (
        np.vstack(all_embs).astype(np.float32),
        all_ids,
        all_names
    )


# =========================================================
# HARD NEGATIVE POOL
# =========================================================

def build_hn_pool(
    embs,
    ids,
    names,
    emb_dim,
    pool_size=10,
):

    id_arr = np.array(ids)

    idx = hnswlib.Index(
        space="cosine",
        dim=emb_dim
    )

    idx.init_index(
        max_elements=len(embs),
        ef_construction=200,
        M=32
    )

    idx.add_items(
        embs,
        list(range(len(embs)))
    )

    idx.set_ef(100)

    labels, _ = idx.knn_query(
        embs,
        k=pool_size * 5 + 1
    )

    pool = {}

    for i, name in enumerate(names):

        hard_negs = []

        for pos in labels[i]:

            if pos == i:
                continue

            if id_arr[pos] == ids[i]:
                continue

            hard_negs.append(
                names[pos]
            )

            if len(hard_negs) == pool_size:
                break

        pool[name] = hard_negs

    return pool


# =========================================================
# QUICK EVAL
# =========================================================

@torch.no_grad()
def quick_eval(
    model,
    query_loader,
    gallery_loader,
    device,
    top_k=15,
):

    model.eval()

    g_embs = []
    g_items = []

    for batch in gallery_loader:

        imgs, item_ids, _, _, _ = batch

        imgs = imgs.to(device)

        emb = model.encode_image(imgs)

        g_embs.append(
            emb.cpu().numpy()
        )

        g_items.extend(item_ids)

    g_embs = np.concatenate(g_embs)

    q_embs = []
    q_items = []

    for batch in query_loader:

        imgs, item_ids, _, _, _ = batch

        imgs = imgs.to(device)

        emb = model.encode_image(imgs)

        q_embs.append(
            emb.cpu().numpy()
        )

        q_items.extend(item_ids)

    q_embs = np.concatenate(q_embs)

    sims = q_embs @ g_embs.T

    top_idx = np.argsort(
        -sims,
        axis=1
    )[:, :top_k]

    retrieved = [
        [g_items[i] for i in row]
        for row in top_idx
    ]

    res = evaluate(
        query_ids=q_items,
        retrieved=retrieved,
        gallery_ids=g_items,
        item_to_imgs={},
        K_values=[5, 10, 15],
    )

    model.train()

    return res


# =========================================================
# TRAIN
# =========================================================

def train():

    args = get_args()

    set_seed(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    root = Path(args.dataset_root)

    img_root = root / "img"

    splits, img_to_item = parse_eval_partition(
        root / "list_eval_partition.txt"
    )

    bbox_map = parse_bboxes(
        root / "list_bbox_inshop.txt"
    )

    train_rows = [

        {
            "image_name": p,
            "item_id": img_to_item[p]
        }

        for p in splits["train"]
    ]

    transform = get_clip_transform(
        image_size=224,
        augment=True
    )

    eval_transform = get_clip_transform(
        image_size=224,
        augment=False
    )

    # -----------------------------------------------------
    # model
    # -----------------------------------------------------

    model = VisualSearchModel(
        alpha=args.alpha,
        unfreeze_last_n=args.unfreeze_last_n,
        embed_dim=args.embed_dim,
    ).to(device)

    # -----------------------------------------------------
    # initial HN pool
    # -----------------------------------------------------

    print("\n[HN] Building initial pool...")

    embs, ids, names = embed_rows(
        train_rows,
        model,
        eval_transform,
        img_root,
        bbox_map,
        device,
    )

    hard_neg_pool = build_hn_pool(
        embs,
        ids,
        names,
        emb_dim=args.embed_dim,
    )

    # -----------------------------------------------------
    # dataset
    # -----------------------------------------------------

    hn_dataset = HardNegTripletDataset(
        rows=train_rows,
        img_root=img_root,
        bbox_map=bbox_map,
        hard_neg_pool=hard_neg_pool,
        transform=transform,
    )

    train_loader = DataLoader(
        hn_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # -----------------------------------------------------
    # eval loaders
    # -----------------------------------------------------

    from src.dataset import DeepFashionDataset

    query_dataset = DeepFashionDataset(
        img_root,
        splits["query"],
        img_to_item,
        bbox_map,
        eval_transform,
        use_gt_bbox=True,
    )

    gallery_dataset = DeepFashionDataset(
        img_root,
        splits["gallery"],
        img_to_item,
        bbox_map,
        eval_transform,
        use_gt_bbox=True,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    # -----------------------------------------------------
    # losses
    # -----------------------------------------------------

    supcon_loss = SupConLoss(
        temperature=args.temperature
    )

    triplet_loss = TripletLoss(
        margin=args.triplet_margin
    )

    # -----------------------------------------------------
    # optimizer
    # -----------------------------------------------------

    optimizer = AdamW(
        model.trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(
        enabled=(device == "cuda")
    )

    best_r10 = 0.0

    history = []

    out_dir = Path(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # EPOCH LOOP
    # =====================================================

    for epoch in range(args.epochs):

        # -------------------------------------------------
        # refresh hard negatives
        # -------------------------------------------------

        if (
            epoch > 0
            and
            epoch % 3 == 0
        ):

            print(
                "\n[HN] Refreshing pool..."
            )

            embs, ids, names = embed_rows(
                train_rows,
                model,
                eval_transform,
                img_root,
                bbox_map,
                device,
            )

            hard_neg_pool = build_hn_pool(
                embs,
                ids,
                names,
                emb_dim=args.embed_dim,
            )

            hn_dataset.hard_neg_pool = (
                hard_neg_pool
            )

        model.train()

        total_loss = 0.0

        total_sup = 0.0

        total_tri = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for anchor, positive, negative in pbar:

            anchor = anchor.to(device)

            positive = positive.to(device)

            negative = negative.to(device)

            optimizer.zero_grad()

            with autocast(
                enabled=(device == "cuda")
            ):

                z_a = model.encode_image(
                    anchor
                )

                z_p = model.encode_image(
                    positive
                )

                z_n = model.encode_image(
                    negative
                )

                embeddings = torch.cat(
                    [z_a, z_p],
                    dim=0
                )

                labels = torch.arange(
                    z_a.shape[0],
                    device=device
                )

                labels = torch.cat(
                    [labels, labels],
                    dim=0
                )

                loss_sup = supcon_loss(
                    embeddings,
                    labels
                )

                loss_tri = triplet_loss(
                    z_a,
                    z_p,
                    z_n
                )

                loss = (
                    loss_sup
                    +
                    args.triplet_weight
                    * loss_tri
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.trainable_params(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

            total_loss += loss.item()

            total_sup += loss_sup.item()

            total_tri += loss_tri.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)

        avg_sup = total_sup / len(train_loader)

        avg_tri = total_tri / len(train_loader)

        print(
            f"\nEpoch {epoch+1}/{args.epochs} | "
            f"Total: {avg_loss:.4f} | "
            f"InfoNCE: {avg_sup:.4f} | "
            f"Triplet: {avg_tri:.4f}"
        )

        # -------------------------------------------------
        # eval
        # -------------------------------------------------

        res = quick_eval(
            model,
            query_loader,
            gallery_loader,
            device,
        )

        r10 = res.recall[10][0]

        print(res)

        history.append({

            "epoch": epoch + 1,

            "loss": avg_loss,

            "recall@10": r10,
        })

        if r10 > best_r10:

            best_r10 = r10

            torch.save({

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_recall":
                    best_r10,

                "args":
                    vars(args),

            }, str(
                out_dir /
                "clip_finetuned_best.pt"
            ))

            print(
                f"\n[Train] "
                f"★ Best Recall@10: "
                f"{best_r10:.4f}"
            )

    with open(
        out_dir / "train_history.json",
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    print(
        f"\nTraining done. "
        f"Best Recall@10: {best_r10:.4f}"
    )


if __name__ == "__main__":
    train()
