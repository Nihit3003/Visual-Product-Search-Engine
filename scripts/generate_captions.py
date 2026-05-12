"""
Final offline BLIP caption generation
for multimodal retrieval fusion.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from PIL import Image

from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    parse_bboxes,
    bbox_crop,
)


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
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


# =========================================================
# CAPTIONER
# =========================================================

class CaptionGenerator:

    PROMPT = (
        "Describe the clothing item including "
        "color, material, fit, texture, "
        "style, pattern, and garment type."
    )

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

        print(
            f"[BLIP-2] Loading "
            f"{model_id}"
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                model_id,
                use_fast=False
            )
        )

        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.model = (
            Blip2ForConditionalGeneration
            .from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
        ).to(self.device)

        self.model.eval()

        for p in self.model.parameters():

            p.requires_grad_(False)

        print(
            "[BLIP-2] Ready"
        )

    # =====================================================
    # GENERATE BATCH
    # =====================================================

    @torch.no_grad()
    def generate_batch(
        self,
        images,
    ):

        inputs = self.processor(
            images=images,
            text=[
                self.PROMPT
            ] * len(images),
            return_tensors="pt",
            padding=True,
        )

        inputs = {

            k: v.to(self.device)

            for k, v in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=3,
        )

        captions = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
        )

        captions = [

            c.strip()

            for c in captions
        ]

        return captions


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_float32_matmul_precision(
        "high"
    )

    dataset_root = Path(
        args.dataset_root
    )

    img_root = dataset_root / "img"

    bbox_map = parse_bboxes(
        dataset_root /
        "list_bbox_inshop.txt"
    )

    all_images = sorted(
        img_root.rglob("*.jpg")
    )

    if args.limit is not None:

        all_images = all_images[
            :args.limit
        ]

    print(
        f"[Data] Images: "
        f"{len(all_images):,}"
    )

    captioner = CaptionGenerator(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )

    results = {}

    batch_imgs = []

    batch_paths = []

    # =====================================================
    # FLUSH
    # =====================================================

    def flush():

        if not batch_imgs:
            return

        try:

            captions = (
                captioner.generate_batch(
                    batch_imgs
                )
            )

            for rel_path, cap in zip(
                batch_paths,
                captions
            ):

                results[rel_path] = cap

        except Exception as e:

            print(
                f"[BLIP-2] "
                f"Batch failed: {e}"
            )

            for rel_path in batch_paths:

                results[rel_path] = ""

        batch_imgs.clear()

        batch_paths.clear()

    # =====================================================
    # LOOP
    # =====================================================

    for path in tqdm(
        all_images,
        desc="Generating captions"
    ):

        try:

            image = Image.open(
                path
            ).convert("RGB")

            rel_path = str(
                path.relative_to(img_root)
            )

            # -------------------------------------------------
            # GT bbox crop
            # -------------------------------------------------

            bbox = bbox_map.get(
                rel_path
            )

            if bbox is not None:

                image = bbox_crop(
                    image,
                    bbox
                )

            batch_imgs.append(
                image
            )

            batch_paths.append(
                rel_path
            )

            # -------------------------------------------------
            # flush
            # -------------------------------------------------

            if (
                len(batch_imgs)
                >=
                args.batch_size
            ):

                flush()

        except Exception as e:

            print(
                f"[ERROR] "
                f"{path.name}: {e}"
            )

    flush()

    # =====================================================
    # SAVE
    # =====================================================

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
        f"\nSaved captions → "
        f"{output_path}"
    )


if __name__ == "__main__":

    main()
