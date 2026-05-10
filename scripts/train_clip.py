"""
Improved CLIP Fine-Tuning Script

Enhancements:
- stronger SupCon training
- hard-negative learning
- AMP acceleration
- gradient accumulation
- better retrieval evaluation
- stable cosine optimization
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.cuda.amp import (
    GradScaler,
    autocast
)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import (
    build_dataloaders,
)

from src.model import (
    VisualSearchModel,
    SupConLoss,
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
        default=15
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=96
    )

    p.add_argument(
        "--lr",
        type=float,
        default=2e-5
    )

    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=0.05
    )

    p.add_argument(
        "--unfreeze_last_n",
        type=int,
        default=4
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
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--eval_every",
        type=int,
        default=1
    )

    p.add_argument(
        "--resume",
        default=None
    )

    p.add_argument(
        "--accum_steps",
        type=int,
        default=2
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    # -----------------------------------------------------
    # gallery embeddings
    # -----------------------------------------------------

    g_embs = []
    g_items = []

    for batch in tqdm(
        gallery_loader,
        desc="  Gallery",
        leave=False
    ):

        imgs, item_ids, _, _, _ = batch

        imgs = imgs.to(device)

        emb = model.encode_image(imgs)

        emb = emb.cpu().numpy()

        g_embs.append(emb)

        g_items.extend(item_ids)

    g_embs = np.concatenate(g_embs, axis=0)

    # -----------------------------------------------------
    # query embeddings
    # -----------------------------------------------------

    q_embs = []
    q_items = []

    for batch in tqdm(
        query_loader,
        desc="  Query",
        leave=False
    ):

        imgs, item_ids, _, _, _ = batch

        imgs = imgs.to(device)

        emb = model.encode_image(imgs)

        emb = emb.cpu().numpy()

        q_embs.append(emb)

        q_items.extend(item_ids)

    q_embs = np.concatenate(q_embs, axis=0)

    # -----------------------------------------------------
    # cosine similarity
    # -----------------------------------------------------

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

    print(
        f"[Train] Device: {device}"
    )

    out_dir = Path(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # loaders
    # -----------------------------------------------------

    loaders = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=224,
        use_gt_bbox=True,
    )

    train_loader = loaders["train"]
    query_loader = loaders["query"]
    gallery_loader = loaders["gallery"]

    # -----------------------------------------------------
    # model
    # -----------------------------------------------------

    model = VisualSearchModel(
        alpha=args.alpha,
        unfreeze_last_n=args.unfreeze_last_n,
        embed_dim=args.embed_dim,
    ).to(device)

    total, trainable = model.param_count()

    print(
        f"[Train] Params: "
        f"{total/1e6:.1f}M total | "
        f"{trainable/1e6:.2f}M trainable"
    )

    # -----------------------------------------------------
    # optimizer
    # -----------------------------------------------------

    criterion = SupConLoss(
        temperature=args.temperature
    )

    optimizer = AdamW(
        model.trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    scaler = GradScaler(
        enabled=(device == "cuda")
    )

    # -----------------------------------------------------
    # resume
    # -----------------------------------------------------

    start_epoch = 0
    best_r10 = 0.0

    if args.resume:

        ckpt = torch.load(
            args.resume,
            map_location=device
        )

        model.load_state_dict(
            ckpt["model_state_dict"],
            strict=False
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )

        start_epoch = ckpt["epoch"] + 1

        best_r10 = ckpt.get(
            "best_recall",
            0.0
        )

        print(
            f"[Train] Resumed "
            f"from epoch {start_epoch}"
        )

    # -----------------------------------------------------
    # history
    # -----------------------------------------------------

    history = {
        "train_loss": [],
        "recall@10": [],
        "ndcg@10": [],
        "mAP@10": [],
    }

    # =====================================================
    # EPOCH LOOP
    # =====================================================

    for epoch in range(
        start_epoch,
        args.epochs
    ):

        model.train()

        epoch_loss = 0.0

        t0 = time.time()

        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for step, batch in enumerate(pbar):

            imgs, item_ids, labels, _, categories = batch

            imgs = imgs.to(device)

            labels = labels.to(device)

            # -------------------------------------------------
            # AMP forward
            # -------------------------------------------------

            with autocast(
                enabled=(device == "cuda")
            ):

                embs = model.encode_image(imgs)

                loss = criterion(
                    embs,
                    labels
                )

                loss = (
                    loss /
                    args.accum_steps
                )

            # -------------------------------------------------
            # backward
            # -------------------------------------------------

            scaler.scale(loss).backward()

            # -------------------------------------------------
            # step
            # -------------------------------------------------

            if (
                (step + 1)
                %
                args.accum_steps
                == 0
            ):

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.trainable_params(),
                    args.grad_clip
                )

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad()

            epoch_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        avg_loss = (
            epoch_loss /
            len(train_loader)
        )

        elapsed = time.time() - t0

        print(
            f"\n[Train] Epoch {epoch+1} | "
            f"Loss={avg_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(
            avg_loss
        )

        # -----------------------------------------------------
        # eval
        # -----------------------------------------------------

        if (
            (epoch + 1)
            %
            args.eval_every
            == 0
        ):

            print(
                "[Train] Running eval..."
            )

            res = quick_eval(
                model,
                query_loader,
                gallery_loader,
                device,
            )

            print(res)

            r10 = res.recall[10][0]

            history["recall@10"].append(r10)
            history["ndcg@10"].append(
                res.ndcg[10][0]
            )
            history["mAP@10"].append(
                res.mAP[10][0]
            )

            # -------------------------------------------------
            # save best
            # -------------------------------------------------

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
                    f"[Train] ★ "
                    f"New best Recall@10: "
                    f"{best_r10:.4f}"
                )

        # -----------------------------------------------------
        # save last
        # -----------------------------------------------------

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
            "clip_finetuned_last.pt"
        ))

    # =====================================================
    # SAVE HISTORY
    # =====================================================

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
        f"\n[Train] Done. "
        f"Best Recall@10 = {best_r10:.4f}"
    )


if __name__ == "__main__":
    train()
