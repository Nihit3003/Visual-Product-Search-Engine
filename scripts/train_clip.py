"""
CLIP Fine-Tuning Script
=======================
Fine-tunes the CLIP vision encoder (last N blocks) on DeepFashion
using Supervised Contrastive Loss (SupCon).

Key design choices:
  - Mixed precision (AMP) for 2× throughput on Kaggle P100/T4
  - Cosine LR schedule with linear warm-up
  - Gradient clipping to stabilise SupCon training
  - Checkpoint every epoch; best model saved by Recall@10 on gallery

Usage (Kaggle):
    python scripts/train_clip.py \
        --dataset_root /kaggle/input/deepfashion-inshop \
        --output_dir   /kaggle/working/checkpoints \
        --epochs       10 \
        --batch_size   128 \
        --lr           1e-4 \
        --unfreeze_last_n 4 \
        --seed         42
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import (
    build_dataloaders,
    get_clip_transform,
    parse_eval_partition,
    parse_bboxes,
)
from src.model   import VisualSearchModel, SupConLoss
from src.metrics import evaluate, MetricResults


# ─────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root",    required=True)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=5e-6)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--temperature",     type=float, default=0.07)
    p.add_argument("--unfreeze_last_n", type=int,   default=4)
    p.add_argument("--embed_dim",       type=int,   default=256)
    p.add_argument("--alpha",           type=float, default=1.0,
                   help="Alpha for fusion during training; 1.0 = vision only")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--warmup_epochs",   type=int,   default=1)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--eval_every",      type=int,   default=1,
                   help="Run quick gallery eval every N epochs")
    p.add_argument("--resume",          default=None, help="Resume from checkpoint path")
    return p.parse_args()


# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
#  Quick in-training evaluation (vision-only, no re-rank)
# ─────────────────────────────────────────────

@torch.no_grad()
def quick_eval(model, query_loader, gallery_loader, device, top_k=15):
    model.eval()
    # Build gallery embeddings
    g_embs, g_items = [], []
    for imgs, item_ids, _, _ in tqdm(gallery_loader, desc="  Gallery emb", leave=False):
        embs = model.encode_image(imgs.to(device)).cpu().numpy()
        g_embs.append(embs)
        g_items.extend(item_ids)
    g_embs = np.concatenate(g_embs, axis=0)

    # Query
    q_embs, q_items = [], []
    for imgs, item_ids, _, _ in tqdm(query_loader, desc="  Query emb", leave=False):
        embs = model.encode_image(imgs.to(device)).cpu().numpy()
        q_embs.append(embs)
        q_items.extend(item_ids)
    q_embs = np.concatenate(q_embs, axis=0)

    # Cosine similarity (dot product on normalised vectors)
    sims = q_embs @ g_embs.T   # (Q, G)
    top_k_indices = np.argsort(-sims, axis=1)[:, :top_k]
    retrieved = [[g_items[i] for i in row] for row in top_k_indices]

    results = evaluate(
        query_ids=q_items,
        retrieved=retrieved,
        gallery_ids=g_items,
        item_to_imgs={},
        K_values=[5, 10, 15],
    )
    model.train()
    return results


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────

def train():
    args   = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device} | Seed: {args.seed}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────
    loaders = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=224,
        use_gt_bbox=True,
    )
    train_loader   = loaders["train"]
    query_loader   = loaders["query"]
    gallery_loader = loaders["gallery"]

    # ── Model ────────────────────────────────
    model = VisualSearchModel(
        alpha=args.alpha,
        unfreeze_last_n=args.unfreeze_last_n,
        embed_dim=args.embed_dim,
    ).to(device)
    total, trainable = model.param_count()
    print(f"[Train] Params: {total/1e6:.1f}M total | {trainable/1e6:.2f}M trainable")

    # ── Loss / Optimizer / Scheduler ─────────
    criterion = SupConLoss(temperature=args.temperature)
    optimizer = AdamW(model.trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps   = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=n_steps,
        pct_start=args.warmup_epochs / args.epochs,
        anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
    )
    scaler = GradScaler(enabled=False)

    # ── Resume ───────────────────────────────
    start_epoch = 0
    best_recall = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_recall = ckpt.get("best_recall", 0.0)
        print(f"[Train] Resumed from epoch {start_epoch}")

    # ── History ──────────────────────────────
    history = {"train_loss": [], "recall@10": [], "ndcg@10": [], "mAP@10": []}

    # ── Epoch loop ───────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (imgs, item_ids, labels, _) in enumerate(pbar):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embs = model.encode_image(imgs)
            loss = criterion(embs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.trainable_params(),
                args.grad_clip
            )
            
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"\n[Train] Epoch {epoch+1} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")
        history["train_loss"].append(avg_loss)

        # ── Eval ─────────────────────────────
        if (epoch + 1) % args.eval_every == 0:
            print("[Train] Running quick eval …")
            res = quick_eval(model, query_loader, gallery_loader, device)
            print(res)
            r10 = res.recall[10][0]
            history["recall@10"].append(r10)
            history["ndcg@10"].append(res.ndcg[10][0])
            history["mAP@10"].append(res.mAP[10][0])

            # Save best
            if r10 > best_recall:
                best_recall = r10
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_recall": best_recall,
                    "args": vars(args),
                }, str(out_dir / "clip_finetuned_best.pt"))
                print(f"[Train] ★ New best Recall@10: {best_recall:.4f}")

        # Always save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_recall": best_recall,
            "args": vars(args),
        }, str(out_dir / "clip_finetuned_last.pt"))

    # ── Save history ─────────────────────────
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] Training complete. Best Recall@10: {best_recall:.4f}")


if __name__ == "__main__":
    train()
