"""
Improved CLIP-based visual retrieval model.

Enhancements:
- stronger SupCon
- hard-negative triplet loss
- projection MLP
- temperature scaling
- improved normalization
- retrieval robustness
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
    Supervised Contrastive Loss
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

        features = F.normalize(
            features,
            dim=-1
        )

        sim = torch.matmul(
            features,
            features.T
        )

        sim = sim / self.temperature

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

        sim_max, _ = torch.max(
            sim,
            dim=1,
            keepdim=True
        )

        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * (1 - self_mask)

        log_prob = sim - torch.log(
            exp_sim.sum(dim=1, keepdim=True) + 1e-8
        )

        pos_count = pos_mask.sum(dim=1)

        mean_log_prob = (
            pos_mask * log_prob
        ).sum(dim=1) / (pos_count + 1e-8)

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
# HARD NEGATIVE TRIPLET LOSS
# =========================================================

class TripletLoss(nn.Module):

    """
    Hard-negative triplet margin loss
    """

    def __init__(
        self,
        margin=0.2
    ):

        super().__init__()

        self.loss_fn = nn.TripletMarginLoss(
            margin=margin,
            p=2
        )

    def forward(
        self,
        anchor,
        positive,
        negative
    ):

        anchor = F.normalize(
            anchor,
            dim=-1
        )

        positive = F.normalize(
            positive,
            dim=-1
        )

        negative = F.normalize(
            negative,
            dim=-1
        )

        return self.loss_fn(
            anchor,
            positive,
            negative
        )


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
            model.text_projection.shape[0]
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
        # projection head
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
    # IMAGE ENCODING
    # =====================================================

    def encode_image(
        self,
        pixel_values: torch.Tensor
    ):

        feat = self.visual(
            pixel_values
        )

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
