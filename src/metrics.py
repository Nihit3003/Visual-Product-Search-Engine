"""
Retrieval evaluation metrics.

Implements:
  - Recall@K
  - NDCG@K  (Normalized Discounted Cumulative Gain)
  - mAP@K   (Mean Average Precision)

All metrics follow the DeepFashion In-Shop protocol:
  Two images are a correct match iff they share the same item_id.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
#  Per-query helpers
# ─────────────────────────────────────────────

def _relevance_list(retrieved_ids: list[str], gt_ids: set[str]) -> list[int]:
    """Binary relevance list: 1 if retrieved item_id ∈ gt_ids, else 0."""
    return [1 if iid in gt_ids else 0 for iid in retrieved_ids]


def recall_at_k(rel: list[int], k: int) -> float:
    """Recall@K: 1 if any of the top-K are relevant, else 0."""
    return 1.0 if sum(rel[:k]) > 0 else 0.0


def ndcg_at_k(rel: list[int], k: int) -> float:
    """
    NDCG@K with binary relevance.
    ideal DCG = sum_{i=1}^{min(|relevant|, k)} 1 / log2(i+1)
    """
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rel[:k]))
    n_rel = min(sum(rel), k)
    idcg  = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(rel: list[int], k: int) -> float:
    """Average Precision@K."""
    hits = 0
    score = 0.0
    for i, r in enumerate(rel[:k]):
        if r:
            hits += 1
            score += hits / (i + 1)
    n_rel = sum(rel)  # total relevant in full gallery
    denom = min(n_rel, k)
    return score / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────
#  Aggregated metrics container
# ─────────────────────────────────────────────

@dataclass
class MetricResults:
    """Stores mean ± std for all metrics at all K values."""
    K_values: list[int] = field(default_factory=lambda: [5, 10, 15])

    recall : dict[int, tuple[float, float]] = field(default_factory=dict)
    ndcg   : dict[int, tuple[float, float]] = field(default_factory=dict)
    mAP    : dict[int, tuple[float, float]] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["=" * 60, f"{'Metric':<20} {'@5':>10} {'@10':>10} {'@15':>10}", "=" * 60]
        for name, d in [("Recall", self.recall), ("NDCG", self.ndcg), ("mAP", self.mAP)]:
            row = f"{name:<20}"
            for k in self.K_values:
                if k in d:
                    m, s = d[k]
                    row += f"  {m:.4f}±{s:.4f}"
                else:
                    row += f"  {'N/A':>12}"
            lines.append(row)
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            f"Recall@{k}": {"mean": self.recall[k][0], "std": self.recall[k][1]}
            for k in self.K_values if k in self.recall
        } | {
            f"NDCG@{k}": {"mean": self.ndcg[k][0], "std": self.ndcg[k][1]}
            for k in self.K_values if k in self.ndcg
        } | {
            f"mAP@{k}": {"mean": self.mAP[k][0], "std": self.mAP[k][1]}
            for k in self.K_values if k in self.mAP
        }


# ─────────────────────────────────────────────
#  Main evaluation function
# ─────────────────────────────────────────────

def evaluate(
    query_ids   : list[str],
    retrieved   : list[list[str]],   # outer: queries, inner: top-K item_ids
    gallery_ids : list[str],
    item_to_imgs: dict[str, list[str]],   # item_id → list of gallery img paths
    K_values    : list[int] = [5, 10, 15],
) -> MetricResults:
    """
    Compute Recall@K, NDCG@K, mAP@K for all queries.

    Args:
        query_ids    : item_id for each query image
        retrieved    : for each query, a list of retrieved item_ids (ordered by rank)
        gallery_ids  : item_id for each gallery image (unused directly)
        item_to_imgs : maps item_id → gallery image paths (to compute |relevant|)
        K_values     : list of K values to evaluate
    Returns:
        MetricResults with per-K mean ± std.
    """
    max_k = max(K_values)
    per_query: dict[str, list[float]] = {
        f"recall@{k}": [] for k in K_values
    } | {
        f"ndcg@{k}": [] for k in K_values
    } | {
        f"map@{k}": [] for k in K_values
    }

    for q_item, ret_items in zip(query_ids, retrieved):
        # Ground truth: all gallery items with the same item_id
        # (exclude the query image itself — DeepFashion protocol keeps query ≠ gallery)
        gt_ids = {q_item}
        rel    = _relevance_list(ret_items[:max_k], gt_ids)

        for k in K_values:
            per_query[f"recall@{k}"].append(recall_at_k(rel, k))
            per_query[f"ndcg@{k}"].append(ndcg_at_k(rel, k))
            per_query[f"map@{k}"].append(ap_at_k(rel, k))

    results = MetricResults(K_values=K_values)
    for k in K_values:
        r = np.array(per_query[f"recall@{k}"])
        n = np.array(per_query[f"ndcg@{k}"])
        m = np.array(per_query[f"map@{k}"])
        results.recall[k] = (float(r.mean()), float(r.std()))
        results.ndcg[k]   = (float(n.mean()), float(n.std()))
        results.mAP[k]    = (float(m.mean()), float(m.std()))

    return results


# ─────────────────────────────────────────────
#  Convenience wrapper for multi-seed evaluation
# ─────────────────────────────────────────────

def evaluate_multi_seed(
    results_per_seed: list[MetricResults],
    K_values: list[int] = [5, 10, 15],
) -> MetricResults:
    """
    Aggregate MetricResults across multiple seeds.
    Mean and std computed over seeds (not queries).
    """
    agg = MetricResults(K_values=K_values)
    for k in K_values:
        rec = [r.recall[k][0] for r in results_per_seed if k in r.recall]
        ndg = [r.ndcg[k][0]   for r in results_per_seed if k in r.ndcg]
        map_ = [r.mAP[k][0]   for r in results_per_seed if k in r.mAP]

        agg.recall[k] = (np.mean(rec), np.std(rec))
        agg.ndcg[k]   = (np.mean(ndg), np.std(ndg))
        agg.mAP[k]    = (np.mean(map_), np.std(map_))
    return agg
