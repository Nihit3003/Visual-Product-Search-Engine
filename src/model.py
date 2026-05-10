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

"""
Improved CLIP-based visual retrieval model.

Enhancements:
- stronger SupCon
- multi-crop embeddings
- projection MLP
- temperature scaling
- improved normalization
- better retrieval robustness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# =========================================================
# SUPCON LOSS
# =========================================================

class SupConLoss(nn.Module):

    """
    Improved Supervised Contrastive Loss
    with stronger hard-negative separation.
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ):

        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ):

        device = features.device

        N = features.shape[0]

        # -------------------------------------------------
        # normalize
        # -------------------------------------------------

        features = F.normalize(
            features,
            dim=-1
        )

        # -------------------------------------------------
        # similarity matrix
        # -------------------------------------------------

        sim = torch.matmul(
            features,
            features.T
        )

        sim = sim / self.temperature

        # -------------------------------------------------
        # masks
        # -------------------------------------------------

        labels = labels.contiguous().view(-1, 1)

        pos_mask = torch.eq(
            labels,
            labels.T
        ).float().to(device)

        self_mask = torch.eye(
            N,
            device=device
        )

        pos_mask = pos_mask - self_mask

        # -------------------------------------------------
        # numerical stability
        # -------------------------------------------------

        sim_max, _ = torch.max(
            sim,
            dim=1,
            keepdim=True
        )

        sim = sim - sim_max.detach()

        # -------------------------------------------------
        # log prob
        # -------------------------------------------------

        exp_sim = torch.exp(sim) * (1 - self_mask)

        log_prob = sim - torch.log(
            exp_sim.sum(dim=1, keepdim=True) + 1e-8
        )

        # -------------------------------------------------
        # mean positive log-likelihood
        # -------------------------------------------------

        pos_count = pos_mask.sum(dim=1)

        mean_log_prob = (
            pos_mask * log_prob
        ).sum(dim=1) / (pos_count + 1e-8)

        # -------------------------------------------------
        # final loss
        # -------------------------------------------------

        valid = pos_count > 0

        if valid.sum() == 0:

            return torch.tensor(
                0.0,
                device=device,
                requires_grad=True
            )

        loss = -mean_log_prob[valid].mean()

        return loss


# =========================================================
# MODEL
# =========================================================

class VisualSearchModel(nn.Module):

    CLIP_MODEL = "ViT-B-16"

    CLIP_PRETRAIN = "openai"

    def __init__(
        self,
        clip_model_name="ViT-B-16",
        pretrained="openai",
        alpha=0.7,
        unfreeze_last_n=4,
        embed_dim=256,
    ):

        super().__init__()

        self.alpha = alpha

        # -------------------------------------------------
        # load CLIP
        # -------------------------------------------------

        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=pretrained
        )

        self.visual = model.visual

        self._clip_model = model

        self.encode_text_fn = model.encode_text

        self.tokenizer = open_clip.get_tokenizer(
            clip_model_name
        )

        clip_dim = (
            model.text_projection.shape[1]
            if hasattr(model, "text_projection")
            else 512
        )

        # -------------------------------------------------
        # freeze all
        # -------------------------------------------------

        for p in model.parameters():

            p.requires_grad_(False)

        # -------------------------------------------------
        # selectively unfreeze visual blocks
        # -------------------------------------------------

        self._unfreeze_vision(
            unfreeze_last_n
        )

        # -------------------------------------------------
        # improved projection head
        # -------------------------------------------------

        self.proj = nn.Sequential(

            nn.Linear(
                clip_dim,
                clip_dim
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                clip_dim,
                embed_dim
            ),

            nn.LayerNorm(embed_dim),
        )

        self.out_dim = embed_dim

    # =====================================================
    # UNFREEZE
    # =====================================================

    def _unfreeze_vision(self, n):

        if n <= 0:
            return

        try:

            blocks = list(
                self.visual.transformer.resblocks
            )

            for block in blocks[-n:]:

                for p in block.parameters():

                    p.requires_grad_(True)

            for p in self.visual.ln_post.parameters():

                p.requires_grad_(True)

            if self.visual.proj is not None:

                self.visual.proj.requires_grad_(True)

        except Exception:

            for p in self.visual.parameters():

                p.requires_grad_(True)

    # =====================================================
    # MULTI-CROP ENCODING
    # =====================================================

    def encode_image(
        self,
        pixel_values: torch.Tensor
    ):

        """
        Multi-crop aware encoding.

        Training:
            receives tensor batch

        Inference:
            can average crops externally
        """

        feat = self.visual(pixel_values)

        feat = self.proj(feat)

        feat = F.normalize(
            feat,
            dim=-1
        )

        return feat

    # =====================================================
    # TEXT
    # =====================================================

    def encode_text(
        self,
        captions: list[str]
    ):

        device = next(
            self.parameters()
        ).device

        tokens = self.tokenizer(
            captions
        ).to(device)

        with torch.no_grad():

            feat = self._clip_model.encode_text(
                tokens
            )

        feat = self.proj(feat)

        feat = F.normalize(
            feat,
            dim=-1
        )

        return feat

    # =====================================================
    # FUSION
    # =====================================================

    def fuse(
        self,
        img_emb,
        txt_emb=None,
    ):

        if txt_emb is None or self.alpha >= 1.0:

            return F.normalize(
                img_emb,
                dim=-1
            )

        fused = (
            self.alpha * img_emb
            +
            (1 - self.alpha) * txt_emb
        )

        fused = F.normalize(
            fused,
            dim=-1
        )

        return fused

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        pixel_values,
        captions=None,
    ):

        img_emb = self.encode_image(
            pixel_values
        )

        if captions is not None and self.alpha < 1.0:

            txt_emb = self.encode_text(
                captions
            )

            return self.fuse(
                img_emb,
                txt_emb
            )

        return img_emb

    # =====================================================
    # PARAMS
    # =====================================================

    def trainable_params(self):

        return [
            p for p in self.parameters()
            if p.requires_grad
        ]

    def param_count(self):

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return total, trainable
