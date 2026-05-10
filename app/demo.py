import argparse
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import (
    get_clip_transform,
    infer_category,
)

from src.index import HNSWIndex

from src.localizer import YOLOLocalizer

from src.model import VisualSearchModel


# =========================================================
# ARGS
# =========================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset_root",
        default="/kaggle/input/deepfashion-inshop"
    )

    p.add_argument(
        "--index_dir",
        default="/kaggle/working/index/condition_C_alpha0.7"
    )

    p.add_argument(
        "--ckpt_path",
        default=None
    )

    p.add_argument(
        "--embed_dim",
        type=int,
        default=256
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.7
    )

    p.add_argument(
        "--top_k",
        type=int,
        default=10
    )

    p.add_argument(
        "--yolo_weights",
        default="yolov8n.pt"
    )

    return p.parse_args()


# =========================================================
# LOADERS
# =========================================================

@st.cache_resource
def load_model(
    ckpt_path,
    alpha,
    embed_dim
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = VisualSearchModel(
        alpha=alpha,
        embed_dim=embed_dim,
        unfreeze_last_n=4
    )

    if (
        ckpt_path
        and
        Path(ckpt_path).exists()
    ):

        state = torch.load(
            ckpt_path,
            map_location=device
        )

        if "model_state_dict" in state:

            state = state["model_state_dict"]

        model.load_state_dict(
            state,
            strict=False
        )

    model.eval().to(device)

    return model, device


@st.cache_resource
def load_index(index_dir):

    return HNSWIndex.load(index_dir)


@st.cache_resource
def load_localizer(yolo_weights):

    try:

        return YOLOLocalizer(
            weights=yolo_weights
        )

    except Exception:

        return None


# =========================================================
# MULTI-CROP EMBEDDING
# =========================================================

def build_embedding(
    img,
    model,
    device
):

    transform = get_clip_transform(
        224,
        augment=False
    )

    w, h = img.size

    full = img

    center = img.crop((
        int(0.15 * w),
        int(0.15 * h),
        int(0.85 * w),
        int(0.85 * h)
    ))

    upper = img.crop((
        0,
        0,
        w,
        int(0.7 * h)
    ))

    crops = [
        full,
        center,
        upper
    ]

    batch = torch.stack([

        transform(c)

        for c in crops

    ]).to(device)

    with torch.no_grad():

        embs = model.encode_image(
            batch
        )

    emb = (
        0.5 * embs[0]
        +
        0.3 * embs[1]
        +
        0.2 * embs[2]
    )

    emb = F.normalize(
        emb,
        dim=-1
    )

    return emb.cpu().numpy()


# =========================================================
# CATEGORY HEURISTIC
# =========================================================

def predict_category(img):

    w, h = img.size

    aspect = h / max(w, 1)

    if aspect > 1.3:
        return "top"

    return "unknown"


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve(
    query_img,
    model,
    index,
    device,
    top_k,
    search_region,
):

    query_category = predict_category(
        query_img
    )

    q_emb = build_embedding(
        query_img,
        model,
        device
    )

    region_map = {

        "Upper Body":
            "upper",

        "Lower Body":
            "lower",

        "Full Outfit":
            None,
    }

    candidates = index.search(

        q_emb,

        top_k=max(
            top_k * 3,
            40
        ),

        query_region=region_map.get(
            search_region,
            None
        ),

        query_category=query_category,
    )

    filtered = []

    for c in candidates:

        path_str = str(
            c["img_path"]
        ).lower()

        category = infer_category(
            path_str
        )

        score = float(
            c["rerank_score"]
        )

        # -------------------------------------------------
        # reranking boosts
        # -------------------------------------------------

        if category == query_category:

            score += 0.10

        caption = c.get(
            "caption",
            ""
        ).lower()

        if any(k in caption for k in [

            "oversized",
            "streetwear",
            "casual",
            "minimal",
            "athletic",

        ]):

            score += 0.015

        c["final_score"] = score

        filtered.append(c)

    filtered = sorted(
        filtered,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return filtered[:top_k]


# =========================================================
# STYLE
# =========================================================

def apply_style():

    st.markdown(
        """
        <style>

        .main {
            background-color: #050816;
            color: white;
        }

        .stApp {
            background-color: #050816;
            color: white;
        }

        .hero-title {
            font-size: 4rem;
            font-weight: 800;
            opacity: 0.15;
        }

        .section-label {
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 2px;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


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
        "Upload a clothing image → "
        "find visually similar products instantly"
    )

    st.markdown("---")

    with st.spinner(
        "Loading models and index..."
    ):

        model, device = load_model(
            args.ckpt_path,
            args.alpha,
            args.embed_dim
        )

        index = load_index(
            args.index_dir
        )

        localizer = load_localizer(
            args.yolo_weights
        )

    left, right = st.columns([1, 2])

    # =====================================================
    # LEFT PANEL
    # =====================================================

    with left:

        st.markdown(
            '<div class="section-label">'
            'QUERY IMAGE'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Upload an image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

        search_region = st.radio(
            "Select clothing region",
            [
                "Upper Body",
                "Lower Body",
                "Full Outfit"
            ]
        )

        if uploaded:

            img = Image.open(
                uploaded
            ).convert("RGB")

            st.image(
                img,
                caption="Original",
                use_container_width=True
            )

            final_crop = img

            # -------------------------------------------------
            # YOLO localization
            # -------------------------------------------------

            if localizer is not None:

                try:

                    detection = localizer.detect(
                        img
                    )

                    conf = detection.get(
                        "confidence",
                        0.0
                    )

                    box = detection.get(
                        "box",
                        None
                    )

                    if box is not None:

                        x1, y1, x2, y2 = box

                        pad_x = int(
                            0.03 * (x2 - x1)
                        )

                        pad_y = int(
                            0.03 * (y2 - y1)
                        )

                        x1 = max(
                            0,
                            x1 + pad_x
                        )

                        y1 = max(
                            0,
                            y1 + pad_y
                        )

                        x2 = min(
                            img.width,
                            x2 - pad_x
                        )

                        y2 = min(
                            img.height,
                            y2 - pad_y
                        )

                        h_box = y2 - y1

                        full_crop = img.crop((
                            x1,
                            y1,
                            x2,
                            y2
                        ))

                        upper_crop = img.crop((
                            x1,
                            y1,
                            x2,
                            int(y1 + 0.55 * h_box)
                        ))

                        lower_crop = img.crop((
                            x1,
                            int(y1 + 0.45 * h_box),
                            x2,
                            y2
                        ))

                        if (
                            search_region
                            ==
                            "Upper Body"
                        ):

                            final_crop = upper_crop

                        elif (
                            search_region
                            ==
                            "Lower Body"
                        ):

                            final_crop = lower_crop

                        else:

                            final_crop = full_crop

                        st.markdown(
                            '<div class="section-label">'
                            'DETECTED CROP'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.image(
                            final_crop,
                            caption=(
                                f"YOLO crop "
                                f"(conf={conf:.2f})"
                            ),
                            use_container_width=True
                        )

                        st.success(
                            f"Detection confidence: "
                            f"{conf:.2f}"
                        )

                    else:

                        st.warning(
                            "No confident detection."
                        )

                except Exception as e:

                    st.warning(
                        f"YOLO failed: {e}"
                    )

            st.session_state[
                "final_crop"
            ] = final_crop

    # =====================================================
    # RIGHT PANEL
    # =====================================================

    with right:

        if (
            uploaded
            and
            "final_crop"
            in st.session_state
        ):

            final_crop = st.session_state[
                "final_crop"
            ]

            st.markdown(
                '<div class="section-label">'
                'RETRIEVAL RESULTS'
                '</div>',
                unsafe_allow_html=True
            )

            t0 = time.time()

            candidates = retrieve(
                final_crop,
                model,
                index,
                device,
                args.top_k,
                search_region,
            )

            latency = (
                time.time() - t0
            ) * 1000

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Results",
                len(candidates)
            )

            m2.metric(
                "Index size",
                f"{len(index):,}"
            )

            m3.metric(
                "Latency",
                f"{latency:.0f} ms"
            )

            st.markdown("---")

            dataset_root = Path(
                args.dataset_root
            )

            rows = [

                candidates[i:i+3]

                for i in range(
                    0,
                    len(candidates),
                    3
                )
            ]

            for row in rows:

                cols = st.columns(3)

                for col, cand in zip(cols, row):

                    with col:

                        img_path = (
                            dataset_root
                            /
                            "img"
                            /
                            cand["img_path"]
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
                            **Similarity:** 
                            {cand['final_score']:.3f}

                            **Item:**  
                            {cand['item_id']}
                            """
                        )

                        caption = cand.get(
                            "caption",
                            ""
                        )

                        if caption:

                            st.caption(
                                caption[:120]
                            )


if __name__ == "__main__":
    main()
