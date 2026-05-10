import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.localizer import YOLOLocalizer


# =========================================================
# ARGS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        required=True,
    )

    parser.add_argument(
        "--output_json",
        required=True,
    )

    parser.add_argument(
        "--model_id",
        default="Salesforce/blip2-flan-t5-xl",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
    )

    return parser.parse_args()


# =========================================================
# CAPTION GENERATOR
# =========================================================

class CaptionGenerator:

    def __init__(
        self,
        model_id,
        max_new_tokens,
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.max_new_tokens = max_new_tokens

        print(f"[BLIP-2] Loading {model_id}")

        self.processor = (
            Blip2Processor.from_pretrained(
                model_id
            )
        )

        self.model = (
            Blip2ForConditionalGeneration
            .from_pretrained(
                model_id,
                torch_dtype=(
                    torch.float16
                    if self.device == "cuda"
                    else torch.float32
                ),
                device_map="auto" if self.device == "cuda" else None,
            )
        )

        self.model.eval()

        print("[BLIP-2] Ready")

    def generate(
        self,
        image,
    ):

        prompt = (
            "Describe the clothing item "
            "including color, style, fit, "
            "and clothing type."
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(
            self.device,
            torch.float16
        )

        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=3,
            )

        caption = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            .strip()
        )

        return caption


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    dataset_root = Path(
        args.dataset_root
    )

    img_root = dataset_root / "img"

    all_images = list(
        img_root.rglob("*.jpg")
    )

    all_images = all_images[:args.limit]

    print(
        f"[Data] Using "
        f"{len(all_images):,} images"
    )

    captioner = CaptionGenerator(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )

    localizer = YOLOLocalizer()

    results = {}

    for path in tqdm(
        all_images,
        desc="Generating captions"
    ):

        try:

            image = Image.open(
                path
            ).convert("RGB")

            detection = localizer.detect(
                image
            )

            crop = detection["cropped"]

            caption = captioner.generate(
                crop
            )

            rel_path = str(
                path.relative_to(img_root)
            )

            results[rel_path] = {
                "caption": caption
            }

        except Exception as e:

            print(
                f"[ERROR] "
                f"{path.name}: {e}"
            )

            continue

    output_path = Path(
        args.output_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"\nSaved captions to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
