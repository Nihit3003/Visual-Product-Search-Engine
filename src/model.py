"""
CLIP-based cross-modal embedding model.

Architecture:
  - Vision encoder  : CLIP ViT (last-N blocks unfrozen for fine-tuning)
  - Text  encoder  : CLIP text encoder (frozen)
  - Fusion         : weighted sum  v = α·φ_V(x̂) + (1-α)·φ_T(c)
  - Output         : L2-normalised 512-dim vector (ViT-B/32)

Loss:
  - Supervised Contrastive Loss (SupCon) — same item_id = positives
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# ─────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss  (Khosla et al., 2020).
    All embeddings from the same item_id are positives for each anchor.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features : (N, D) L2-normalised embeddings
        labels   : (N,)   integer class labels
        """
        device = features.device
        N = features.shape[0]

        # Cosine similarity matrix
        sim = torch.mm(features, features.T) / self.temperature   # (N, N)

        # Mask: positive pairs share the same label (exclude self)
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float().to(device)
        pos_mask.fill_diagonal_(0)

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)
        # Exclude self from denominator
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        denom = exp_sim.masked_fill(self_mask, 0).sum(dim=1, keepdim=True)

        log_prob = sim - torch.log(denom + 1e-8)

        # Mean log-prob over positives
        n_positives = pos_mask.sum(dim=1)
        loss = -(pos_mask * log_prob).sum(dim=1)
        # Only compute loss for anchors that have at least one positive
        valid = n_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = (loss[valid] / n_positives[valid]).mean()
        return loss


# ─────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────

class VisualSearchModel(nn.Module):
    """
    Cross-modal CLIP model for visual product search.

    Args:
        clip_model_name  : open_clip model name (default 'ViT-B-32')
        pretrained       : open_clip pretrained weights tag
        alpha            : image-text fusion weight ∈ [0,1]
        unfreeze_last_n  : number of vision transformer blocks to unfreeze
        embed_dim        : projection head output dim (None = use raw CLIP dim)
    """

    CLIP_MODEL    = "ViT-B-32"
    CLIP_PRETRAIN = "openai"

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        alpha: float = 0.7,
        unfreeze_last_n: int = 4,
        embed_dim: int | None = 256,
    ):
        super().__init__()
        self.alpha = alpha

        # ── Load CLIP ──────────────────────────
        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.visual    = model.visual           # vision encoder
        self.encode_text_fn = model.encode_text
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        clip_dim = model.text_projection.shape[1] if hasattr(model, "text_projection") \
                   else 512

        # ── Freeze text encoder entirely ───────
        for p in model.parameters():
            p.requires_grad_(False)
        # Unfreeze vision encoder selectively
        self._unfreeze_vision(unfreeze_last_n)

        # ── Optional projection head ────────────
        self.proj = None
        if embed_dim is not None and embed_dim != clip_dim:
            self.proj = nn.Sequential(
                nn.Linear(clip_dim, embed_dim, bias=False),
                nn.LayerNorm(embed_dim),
            )
            self.out_dim = embed_dim
        else:
            self.out_dim = clip_dim

        self._clip_model = model   # keep reference for text encoding

    # ── Private helpers ──────────────────────

    def _unfreeze_vision(self, n: int):
        """Unfreeze the last n transformer blocks of the CLIP vision encoder."""
        if n == 0:
            return
        # open_clip ViT: visual.transformer.resblocks is a ModuleList
        try:
            blocks = list(self.visual.transformer.resblocks)
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            # Also unfreeze the final layer norm
            for p in self.visual.ln_post.parameters():
                p.requires_grad_(True)
            # And the projection matrix if it exists
            if self.visual.proj is not None:
                self.visual.proj.requires_grad_(True)
        except AttributeError:
            # Fallback: unfreeze everything in visual
            for p in self.visual.parameters():
                p.requires_grad_(True)

    # ── Forward ──────────────────────────────

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised image embedding."""
        feat = self.visual(pixel_values)
        if self.proj is not None:
            feat = self.proj(feat)
        return F.normalize(feat, dim=-1)

    def encode_text(self, captions: list[str]) -> torch.Tensor:
        """Returns L2-normalised text embedding for a list of captions."""
        device = next(self.parameters()).device
        tokens = self.tokenizer(captions).to(device)
        with torch.no_grad():
            feat = self._clip_model.encode_text(tokens)
        if self.proj is not None:
            feat = self.proj(feat)
        return F.normalize(feat, dim=-1)

    def fuse(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fuse image and text embeddings:
            v = α·img + (1-α)·txt,  then L2-normalise.
        If txt_emb is None (vision-only, α=1), returns img_emb directly.
        """
        if txt_emb is None or self.alpha >= 1.0:
            return img_emb
        fused = self.alpha * img_emb + (1 - self.alpha) * txt_emb
        return F.normalize(fused, dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        captions: list[str] | None = None,
    ) -> torch.Tensor:
        img_emb = self.encode_image(pixel_values)
        if captions is not None and self.alpha < 1.0:
            txt_emb = self.encode_text(captions)
            return self.fuse(img_emb, txt_emb)
        return img_emb

    # ── Utility ──────────────────────────────

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
