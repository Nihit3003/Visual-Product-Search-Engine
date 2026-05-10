"""
Stable BLIP-2 + ITM wrapper.

Fixes:
- transformers compatibility
- safer FP16 handling
- stable batch inference
- stronger caption prompts
- robust fallback behavior
"""

from __future__ import annotations

import torch
from PIL import Image

from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
)

from transformers import (
    BlipForImageTextRetrieval,
    BlipProcessor,
)


# =========================================================
# CAPTIONER
# =========================================================

class FashionCaptioner:

    PROMPT = (
        "Describe the clothing item including "
        "color, material, fit, texture, style, "
        "and garment type."
    )

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-flan-t5-xl",
        device: str = "cuda",
        use_fp16: bool = True,
    ):

        self.device = device

        dtype = (
            torch.float16
            if (
                use_fp16
                and
                device == "cuda"
            )
            else torch.float32
        )

        print(
            f"[BLIP-2] Loading captioner: "
            f"{model_name}"
        )

        # -------------------------------------------------
        # processor
        # -------------------------------------------------

        self.processor = (
            Blip2Processor.from_pretrained(
                model_name
            )
        )

        # -------------------------------------------------
        # model
        # -------------------------------------------------

        self.model = (
            Blip2ForConditionalGeneration
            .from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        ).to(device)

        self.model.eval()

        for p in self.model.parameters():

            p.requires_grad_(False)

        print(
            "[BLIP-2] Captioner ready."
        )

    # =====================================================
    # CAPTION
    # =====================================================

    @torch.no_grad()
    def caption(
        self,
        images: list[Image.Image] | Image.Image,
        batch_size: int = 4,
    ) -> list[str]:

        if isinstance(images, Image.Image):

            images = [images]

        outputs = []

        for i in range(
            0,
            len(images),
            batch_size
        ):

            batch = images[
                i : i + batch_size
            ]

            try:

                inputs = self.processor(
                    images=batch,
                    text=[
                        self.PROMPT
                    ] * len(batch),
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_beams=3,
                )

                decoded = (
                    self.processor.batch_decode(
                        generated,
                        skip_special_tokens=True
                    )
                )

                decoded = [
                    d.strip()
                    for d in decoded
                ]

                outputs.extend(decoded)

            except Exception as e:

                print(
                    f"[BLIP-2] Batch failed: {e}"
                )

                outputs.extend(
                    [""] * len(batch)
                )

        return outputs


# =========================================================
# ITM RERANKER
# =========================================================

class ITMReranker:

    def __init__(
        self,
        model_name: str = "Salesforce/blip-itm-base-coco",
        device: str = "cuda",
        use_fp16: bool = True,
    ):

        self.device = device

        dtype = (
            torch.float16
            if (
                use_fp16
                and
                device == "cuda"
            )
            else torch.float32
        )

        print(
            f"[BLIP-ITM] Loading "
            f"{model_name}"
        )

        self.processor = (
            BlipProcessor.from_pretrained(
                model_name
            )
        )

        self.model = (
            BlipForImageTextRetrieval
            .from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        ).to(device)

        self.model.eval()

        for p in self.model.parameters():

            p.requires_grad_(False)

        print(
            "[BLIP-ITM] Ready."
        )

    # =====================================================
    # SCORE
    # =====================================================

    @torch.no_grad()
    def score(
        self,
        query_image: Image.Image,
        captions: list[str],
        batch_size: int = 8,
    ) -> list[float]:

        scores = []

        for i in range(
            0,
            len(captions),
            batch_size
        ):

            caps = captions[
                i : i + batch_size
            ]

            try:

                inputs = self.processor(
                    images=[
                        query_image
                    ] * len(caps),
                    text=caps,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                out = self.model(
                    **inputs,
                    use_itm_head=True
                )

                probs = (
                    out.itm_score
                    .softmax(dim=-1)[:, 1]
                )

                scores.extend(
                    probs.float()
                    .cpu()
                    .tolist()
                )

            except Exception as e:

                print(
                    f"[BLIP-ITM] "
                    f"Batch failed: {e}"
                )

                scores.extend(
                    [0.0] * len(caps)
                )

        return scores

    # =====================================================
    # RERANK
    # =====================================================

    def rerank(
        self,
        query_image: Image.Image,
        candidates: list[dict],
        caption_key: str = "caption",
    ) -> list[dict]:

        captions = [

            c.get(
                caption_key,
                ""
            )

            for c in candidates
        ]

        scores = self.score(
            query_image,
            captions
        )

        for cand, sc in zip(
            candidates,
            scores
        ):

            cand["itm_score"] = sc

        return sorted(
            candidates,
            key=lambda x: x["itm_score"],
            reverse=True
        )
