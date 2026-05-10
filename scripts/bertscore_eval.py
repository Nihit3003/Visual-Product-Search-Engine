"""
Robust BERTScore evaluation for fashion captions.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from bert_score import score

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# ARGS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gt_json",
        required=True,
    )

    parser.add_argument(
        "--pred_json",
        required=True,
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--model_type",
        default="microsoft/deberta-xlarge-mnli",
    )

    return parser.parse_args()


# =========================================================
# CLEAN
# =========================================================

def clean_text(x):

    if not isinstance(x, str):

        return ""

    x = x.strip()

    x = " ".join(x.split())

    return x


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[Device] {device}"
    )

    # -----------------------------------------------------
    # load GT
    # -----------------------------------------------------

    print(
        "[Load] Ground-truth captions"
    )

    with open(args.gt_json) as f:

        gt = json.load(f)

    # -----------------------------------------------------
    # load predictions
    # -----------------------------------------------------

    print(
        "[Load] Predicted captions"
    )

    with open(args.pred_json) as f:

        pred = json.load(f)

    refs = []

    cands = []

    # -----------------------------------------------------
    # matching
    # -----------------------------------------------------

    for path, pred_data in tqdm(
        pred.items(),
        desc="Matching captions"
    ):

        if path not in gt:
            continue

        gt_caption = clean_text(
            gt[path]
        )

        if isinstance(pred_data, dict):

            pred_caption = clean_text(
                pred_data.get(
                    "caption",
                    ""
                )
            )

        else:

            pred_caption = clean_text(
                pred_data
            )

        if not gt_caption:
            continue

        if not pred_caption:
            continue

        refs.append(gt_caption)

        cands.append(pred_caption)

    print(
        f"\nMatched samples: "
        f"{len(refs):,}"
    )

    if len(refs) == 0:

        raise RuntimeError(
            "No valid caption pairs found."
        )

    # -----------------------------------------------------
    # bertscore
    # -----------------------------------------------------

    print(
        "\nRunning BERTScore..."
    )

    P, R, F1 = score(
        cands,
        refs,
        lang="en",
        model_type=args.model_type,
        batch_size=args.batch_size,
        verbose=True,
        device=device,
    )

    # -----------------------------------------------------
    # aggregate
    # -----------------------------------------------------

    results = {

        "precision": float(
            P.mean().item()
        ),

        "recall": float(
            R.mean().item()
        ),

        "f1": float(
            F1.mean().item()
        ),

        "precision_std": float(
            P.std().item()
        ),

        "recall_std": float(
            R.std().item()
        ),

        "f1_std": float(
            F1.std().item()
        ),

        "num_samples": len(refs),

        "model_type": args.model_type,
    }

    # -----------------------------------------------------
    # print
    # -----------------------------------------------------

    print(
        "\n========== BERTScore =========="
    )

    print(
        f"Precision : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{results['f1']:.4f}"
    )

    print(
        f"Samples   : "
        f"{results['num_samples']:,}"
    )

    # -----------------------------------------------------
    # save
    # -----------------------------------------------------

    output_path = Path(
        args.output_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"\nSaved → {output_path}"
    )


if __name__ == "__main__":

    main()
