"""
Final Streamlit Demo
ViT-L-14 Visual Product Search
"""

import argparse
import sys
import time
from pathlib import Path

import streamlit as st

import torch
import torch.nn.functional as F

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from src.dataset import (
    get_clip_transform
)

from src.index import (
    HNSWIndex
)

from src.localizer import (
    YOLOLocalizer
)

from src.model import (
    VisualSearchModel
)


# =========================================================
# ARGS
# =========================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset_root",
        required=True,
    )

    p.add_argument(
        "--index_dir",
        required=True,
    )

    p.add_argument(
        "--ckpt_path",
        required=True,
    )

    p.add_argument(
        "--embed_dim",
        type=int,
        default=768,
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.7,
    )

    p.add_argument(
        "--top_k",
        type=int,
        default=12,
    )

    p.add_argument(
        "--yolo_weights",
        default="yolov8n.pt",
    )

    return p.parse_args()


# =========================================================
# STYLE
# =========================================================

def apply_style():

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #050816;
            color: white;
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            opacity: 0.18;
            margin-bottom: 0.4rem;
        }

        .metric-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 0.5rem;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model(
    ckpt_path,
    alpha,
    embed_dim,
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = VisualSearchModel(
        clip_model_name="ViT-L-14",
        pretrained="openai",
        alpha=alpha,
        embed_dim=embed_dim,
        unfreeze_last_n=6,
    )

    state = torch.load(
        ckpt_path,
        map_location=device
    )

    if "model_state_dict" in state:

        state = state[
            "model_state_dict"
        ]

    model.load_state_dict(
        state,
        strict=False
    )

    model.eval().to(device)

    return model, device


# =========================================================
# INDEX
# =========================================================

@st.cache_resource
def load_index(index_dir):

    return HNSWIndex.load(
        index_dir
    )


# =========================================================
# YOLO
# =========================================================

@st.cache_resource
def load_localizer(weights):

    try:

        return YOLOLocalizer(
            weights=weights
        )

    except Exception:

        return None


# =========================================================
# MULTI-CROP EMBEDDING
# =========================================================

@torch.no_grad()
def build_embedding(
    image,
    model,
    device,
):

    transform = get_clip_transform(
        image_size=224,
        augment=False
    )

    w, h = image.size

    full = image

    center = image.crop((
        int(0.15 * w),
        int(0.15 * h),
        int(0.85 * w),
        int(0.85 * h)
    ))

    upper = image.crop((
        0,
        0,
        w,
        int(0.7 * h)
    ))

    imgs = [

        transform(full),

        transform(center),

        transform(upper),
    ]

    imgs = torch.stack(
        imgs
    ).to(device)

    emb = model.encode_image(
        imgs
    )

    emb = (
        0.5 * emb[0]
        +
        0.3 * emb[1]
        +
        0.2 * emb[2]
    )

    emb = F.normalize(
        emb,
        dim=-1
    )

    return emb.cpu().numpy()


# =========================================================
# SEARCH
# =========================================================

def retrieve(
    query_img,
    model,
    index,
    device,
    top_k,
):

    q_emb = build_embedding(
        query_img,
        model,
        device,
    )

    results = index.search(
        q_emb,
        top_k=top_k,
        deduplicate_items=True,
    )

    return results


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    st.set_page_config(
        page_title="Visual Product Search",
        layout="wide"
    )

    apply_style()

    st.markdown(
        '<div class="hero-title">'
        'Visual Product Search'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Upload a clothing image to retrieve "
        "visually similar products."
    )

    st.markdown("---")

    # =====================================================
    # LOAD
    # =====================================================

    with st.spinner(
        "Loading models..."
    ):

        model, device = load_model(
            args.ckpt_path,
            args.alpha,
            args.embed_dim,
        )

        index = load_index(
            args.index_dir
        )

        localizer = load_localizer(
            args.yolo_weights
        )

    left, right = st.columns([1, 2])

    # =====================================================
    # LEFT
    # =====================================================

    with left:

        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:

            image = Image.open(
                uploaded
            ).convert("RGB")

            st.image(
                image,
                caption="Original",
                use_container_width=True
            )

            crop = image

            # -------------------------------------------------
            # YOLO localization
            # -------------------------------------------------

            if localizer is not None:

                try:

                    det = localizer.detect(
                        image
                    )

                    crop = det["cropped"]

                    conf = det.get(
                        "confidence",
                        0.0
                    )

                    st.image(
                        crop,
                        caption=(
                            f"Localized "
                            f"(conf={conf:.2f})"
                        ),
                        use_container_width=True
                    )

                except Exception as e:

                    st.warning(
                        f"YOLO failed: {e}"
                    )

            st.session_state[
                "query_crop"
            ] = crop

    # =====================================================
    # RIGHT
    # =====================================================

    with right:

        if (
            uploaded
            and
            "query_crop"
            in st.session_state
        ):

            crop = st.session_state[
                "query_crop"
            ]

            t0 = time.time()

            results = retrieve(
                crop,
                model,
                index,
                device,
                args.top_k,
            )

            latency = (
                time.time() - t0
            ) * 1000

            st.success(
                f"Retrieved "
                f"{len(results)} results "
                f"in {latency:.0f} ms"
            )

            st.markdown("---")

            dataset_root = Path(
                args.dataset_root
            )

            rows = [

                results[i:i+3]

                for i in range(
                    0,
                    len(results),
                    3
                )
            ]

            for row in rows:

                cols = st.columns(3)

                for col, r in zip(cols, row):

                    with col:

                        img_path = (
                            dataset_root
                            / "img"
                            / r["img_path"]
                        )

                        try:

                            result_img = Image.open(
                                img_path
                            ).convert("RGB")

                            st.image(
                                result_img,
                                use_container_width=True
                            )

                        except Exception:

                            st.warning(
                                f"Missing:\n{img_path}"
                            )

                        st.markdown(
                            f"""
                            <div class="metric-card">

                            <b>Similarity:</b>
                            {r['score']:.4f}

                            <br>

                            <b>Item:</b>
                            {r['item_id']}

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        caption = r.get(
                            "caption",
                            ""
                        )

                        if caption:

                            st.caption(
                                caption[:140]
                            )

    st.markdown("---")

    st.caption(
        "ViT-L-14 • Hard Negative Fine-Tuning "
        "• HNSW Retrieval • BLIP Fusion"
    )


if __name__ == "__main__":

    main()
