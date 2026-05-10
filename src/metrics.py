"""
Retrieval evaluation metrics.
"""

from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    field
)

import numpy as np


# =========================================================
# RELEVANCE
# =========================================================

def _relevance_list(
    retrieved_ids: list[str],
    gt_ids: set[str]
):

    return [
        1 if iid in gt_ids else 0
        for iid in retrieved_ids
    ]


# =========================================================
# RECALL
# =========================================================

def recall_at_k(
    rel,
    k
):

    return (
        1.0
        if sum(rel[:k]) > 0
        else 0.0
    )


# =========================================================
# NDCG
# =========================================================

def ndcg_at_k(
    rel,
    k
):

    dcg = sum(

        r / math.log2(i + 2)

        for i, r in enumerate(rel[:k])

    )

    n_rel = min(
        sum(rel),
        k
    )

    idcg = sum(

        1.0 / math.log2(i + 2)

        for i in range(n_rel)

    )

    return (
        dcg / idcg
        if idcg > 0
        else 0.0
    )


# =========================================================
# AP
# =========================================================

def ap_at_k(
    rel,
    k
):

    hits = 0

    score = 0.0

    for i, r in enumerate(rel[:k]):

        if r:

            hits += 1

            score += hits / (i + 1)

    n_rel = sum(rel)

    denom = min(
        n_rel,
        k
    )

    return (
        score / denom
        if denom > 0
        else 0.0
    )


# =========================================================
# METRIC RESULTS
# =========================================================

@dataclass
class MetricResults:

    K_values: list[int] = field(
        default_factory=lambda: [5, 10, 15]
    )

    recall: dict = field(
        default_factory=dict
    )

    ndcg: dict = field(
        default_factory=dict
    )

    mAP: dict = field(
        default_factory=dict
    )

    # -----------------------------------------------------
    # string
    # -----------------------------------------------------

    def __str__(self):

        lines = [

            "=" * 60,

            f"{'Metric':<20}"
            f"{'@5':>14}"
            f"{'@10':>14}"
            f"{'@15':>14}",

            "=" * 60,
        ]

        for name, d in [

            ("Recall", self.recall),

            ("NDCG", self.ndcg),

            ("mAP", self.mAP),

        ]:

            row = f"{name:<20}"

            for k in self.K_values:

                if k in d:

                    mean, std = d[k]

                    row += (
                        f"{mean:.4f}±{std:.4f}"
                    ).rjust(14)

                else:

                    row += "N/A".rjust(14)

            lines.append(row)

        lines.append("=" * 60)

        return "\n".join(lines)

    # -----------------------------------------------------
    # dict
    # -----------------------------------------------------

    def to_dict(self):

        out = {}

        for k in self.K_values:

            if k in self.recall:

                out[f"Recall@{k}"] = {

                    "mean":
                        self.recall[k][0],

                    "std":
                        self.recall[k][1],
                }

            if k in self.ndcg:

                out[f"NDCG@{k}"] = {

                    "mean":
                        self.ndcg[k][0],

                    "std":
                        self.ndcg[k][1],
                }

            if k in self.mAP:

                out[f"mAP@{k}"] = {

                    "mean":
                        self.mAP[k][0],

                    "std":
                        self.mAP[k][1],
                }

        return out


# =========================================================
# MAIN EVALUATION
# =========================================================

def evaluate(
    query_ids,
    retrieved,
    gallery_ids,
    item_to_imgs,
    K_values=[5, 10, 15],
):

    max_k = max(K_values)

    per_query = {

        f"recall@{k}": []

        for k in K_values

    } | {

        f"ndcg@{k}": []

        for k in K_values

    } | {

        f"map@{k}": []

        for k in K_values

    }

    for q_item, ret_items in zip(
        query_ids,
        retrieved
    ):

        gt_ids = {q_item}

        rel = _relevance_list(
            ret_items[:max_k],
            gt_ids
        )

        for k in K_values:

            per_query[
                f"recall@{k}"
            ].append(
                recall_at_k(rel, k)
            )

            per_query[
                f"ndcg@{k}"
            ].append(
                ndcg_at_k(rel, k)
            )

            per_query[
                f"map@{k}"
            ].append(
                ap_at_k(rel, k)
            )

    results = MetricResults(
        K_values=K_values
    )

    for k in K_values:

        r = np.array(
            per_query[f"recall@{k}"]
        )

        n = np.array(
            per_query[f"ndcg@{k}"]
        )

        m = np.array(
            per_query[f"map@{k}"]
        )

        results.recall[k] = (
            float(r.mean()),
            float(r.std())
        )

        results.ndcg[k] = (
            float(n.mean()),
            float(n.std())
        )

        results.mAP[k] = (
            float(m.mean()),
            float(m.std())
        )

    return results


# =========================================================
# MULTI-SEED
# =========================================================

def evaluate_multi_seed(
    results_per_seed,
    K_values=[5, 10, 15],
):

    agg = MetricResults(
        K_values=K_values
    )

    for k in K_values:

        rec = [

            r.recall[k][0]

            for r in results_per_seed

            if k in r.recall

        ]

        ndg = [

            r.ndcg[k][0]

            for r in results_per_seed

            if k in r.ndcg

        ]

        map_ = [

            r.mAP[k][0]

            for r in results_per_seed

            if k in r.mAP

        ]

        agg.recall[k] = (
            float(np.mean(rec)),
            float(np.std(rec))
        )

        agg.ndcg[k] = (
            float(np.mean(ndg)),
            float(np.std(ndg))
        )

        agg.mAP[k] = (
            float(np.mean(map_)),
            float(np.std(map_))
        )

    return agg
