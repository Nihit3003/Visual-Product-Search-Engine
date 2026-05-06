"""
BLIP-2 wrapper for:
  1. Caption generation  (offline indexing Step 2)
  2. Image-Text Matching (ITM) re-ranking (online query Step 4)

Both BLIP-2 components remain FROZEN throughout the project.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipForImageTextRetrieval, BlipProcessor


# ─────────────────────────────────────────────
#  Captioner  (BLIP-2 OPT-2.7B)
# ─────────────────────────────────────────────

class FashionCaptioner:
    """
    Generates fashion-specific captions using BLIP-2.
    Uses a structured prompt to elicit: colour, fit, material, style.

    Args:
        model_name : HuggingFace model name for BLIP-2
        device     : 'cuda' | 'cpu'
        use_fp16   : Load in half precision to save VRAM
    """

    PROMPT = (
        "Question: Describe this clothing item's color, fit, material, "
        "and style in one sentence. Answer:"
    )

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.device = device
        dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

        print(f"[BLIP-2] Loading captioner: {model_name} …")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad_(False)
        print("[BLIP-2] Captioner ready.")

    @torch.no_grad()
    def caption(
        self,
        images: list[Image.Image] | Image.Image,
        batch_size: int = 8,
    ) -> list[str]:
        """
        Generate captions for a list of PIL images.
        Returns a list of caption strings.
        """
        if isinstance(images, Image.Image):
            images = [images]

        captions = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(
                images=batch,
                text=[self.PROMPT] * len(batch),
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            generated = self.model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                temperature=1.0,
            )
            for ids in generated:
                text = self.processor.decode(ids, skip_special_tokens=True).strip()
                # Strip prompt if echoed
                if "Answer:" in text:
                    text = text.split("Answer:")[-1].strip()
                captions.append(text)

        return captions


# ─────────────────────────────────────────────
#  ITM Re-ranker  (BLIP ITM)
# ─────────────────────────────────────────────

class ITMReranker:
    """
    Re-ranks retrieved candidates using BLIP Image-Text Matching score.

    For each (query_image, candidate_caption) pair, returns a scalar ITM
    score in [0, 1]. Candidates are sorted descending by this score.

    Note: Uses the lighter BLIP-base ITM model (not BLIP-2) for speed.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-itm-base-coco",
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.device = device
        dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

        print(f"[BLIP ITM] Loading re-ranker: {model_name} …")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        print("[BLIP ITM] Re-ranker ready.")

    @torch.no_grad()
    def score(
        self,
        query_image: Image.Image,
        captions: list[str],
        batch_size: int = 16,
    ) -> list[float]:
        """
        Returns ITM scores (list of floats) for each (query_image, caption) pair.
        """
        scores = []
        for i in range(0, len(captions), batch_size):
            batch_caps = captions[i : i + batch_size]
            inputs = self.processor(
                images=[query_image] * len(batch_caps),
                text=batch_caps,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(self.device)

            out = self.model(**inputs, use_itm_head=True)
            # out.itm_score : (B, 2)  logits for [non-match, match]
            itm_probs = out.itm_score.softmax(dim=-1)[:, 1]  # match probability
            scores.extend(itm_probs.float().cpu().tolist())

        return scores

    def rerank(
        self,
        query_image: Image.Image,
        candidates: list[dict],
        caption_key: str = "caption",
    ) -> list[dict]:
        """
        Rerank a list of candidate dicts by ITM score.

        Each candidate must have a field `caption_key`.
        Returns sorted list (descending ITM score) with added 'itm_score' field.
        """
        captions = [c[caption_key] for c in candidates]
        scores   = self.score(query_image, captions)
        for cand, sc in zip(candidates, scores):
            cand["itm_score"] = sc
        return sorted(candidates, key=lambda x: x["itm_score"], reverse=True)
