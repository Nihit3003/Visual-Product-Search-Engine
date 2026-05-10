import argparse
import json
import sys
from pathlib import Path

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

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    print(
        "[Load] Ground-truth descriptions"
    )

    with open(args.gt_json) as f:

        gt = json.load(f)

    print(
        "[Load] Predicted captions"
    )

    with open(args.pred_json) as f:

        pred = json.load(f)

    refs = []
    cands = []

    for path, pred_data in tqdm(
        pred.items(),
        desc="Matching captions"
    ):

        if path not in gt:
            continue

        gt_caption = gt[path]

        pred_caption = pred_data.get(
            "caption",
            ""
        )

        if not isinstance(
            gt_caption,
            str
        ):
            continue

        if not isinstance(
            pred_caption,
            str
        ):
            continue

        gt_caption = gt_caption.strip()
        pred_caption = pred_caption.strip()

        if len(gt_caption) == 0:
            continue

        if len(pred_caption) == 0:
            continue

        refs.append(gt_caption)
        cands.append(pred_caption)

    print(
        f"\nMatched samples: "
        f"{len(refs):,}"
    )

    print(
        "\nRunning BERTScore..."
    )

    P, R, F1 = score(
        cands,
        refs,
        lang="en",
        verbose=True
    )

    results = {

        "precision": (
            P.mean().item()
        ),

        "recall": (
            R.mean().item()
        ),

        "f1": (
            F1.mean().item()
        ),

        "num_samples": len(refs)
    }

    print("\nBERTScore Results")

    print(
        f"\nPrecision : "
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
        f"\nSaved results to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
