import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import VisualSearchModel
from src.index import HNSWIndex
from src.localizer import YOLOLocalizer
from src.dataset import get_clip_transform


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
        default="/kaggle/working/index/condition_C_alpha0.6"
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
        default=0.6
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
def load_model(ckpt_path, alpha, embed_dim):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VisualSearchModel(
        alpha=alpha,
        embed_dim=embed_dim,
        unfreeze_last_n=2
    )

    if ckpt_path and Path(ckpt_path).exists():

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
        return YOLOLocalizer(weights=yolo_weights)

    except Exception:
        return None


# =========================================================
# MULTI CROP EMBEDDING
# =========================================================

def build_embedding(img, model, device):

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

    imgs = [full, center, upper]

    embs = []

    with torch.no_grad():

        for im in imgs:

            tensor = transform(im).unsqueeze(0).to(device)

            emb = model.encode_image(tensor)

            emb = emb.cpu().numpy()[0]

            embs.append(emb)

    final_emb = np.mean(embs, axis=0)

    final_emb = final_emb / np.linalg.norm(final_emb)

    return final_emb


# =========================================================
# CATEGORY HEURISTIC
# =========================================================

def predict_category(img):

    w, h = img.size

    aspect = h / w

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
    top_k
):

    query_category = predict_category(query_img)

    q_emb = build_embedding(
        query_img,
        model,
        device
    )

    candidates = index.search(
        q_emb,
        top_k=50
    )

    filtered = []

    for c in candidates:

        path_str = str(c["img_path"]).lower()

        category = "unknown"

        if any(k in path_str for k in [
            "tee",
            "shirt",
            "top",
            "blouse",
            "tank"
        ]):
            category = "top"

        elif any(k in path_str for k in [
            "short",
            "pant",
            "jean",
            "skirt"
        ]):
            category = "bottom"

        c["category"] = category

        score = float(c["score"])

        # lightweight reranking

        if category == query_category:
            score += 0.08

        c["rerank_score"] = score

        filtered.append(c)

    filtered = sorted(
        filtered,
        key=lambda x: x["rerank_score"],
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
        '<div class="hero-title">Visual Product Search</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Upload a clothing image → find visually similar products instantly"
    )

    st.markdown("---")

    with st.spinner("Loading models and index..."):

        model, device = load_model(
            args.ckpt_path,
            args.alpha,
            args.embed_dim
        )

        index = load_index(args.index_dir)

        localizer = load_localizer(
            args.yolo_weights
        )

    left, right = st.columns([1, 2])

    with left:

        st.markdown(
            '<div class="section-label">QUERY IMAGE</div>',
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
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

            img = Image.open(uploaded).convert("RGB")

            st.image(
                img,
                caption="Original",
                width="stretch"
            )

            # ==========================================
            # YOLO
            # ==========================================

            if localizer is not None:

                detection = localizer.detect(img)
            
                conf = detection["confidence"]
            
                box = detection["box"]
            
                if box is not None:
            
                    x1, y1, x2, y2 = box
            
                    # tighter crop padding
            
                    pad_x = int(0.03 * (x2 - x1))
                    pad_y = int(0.03 * (y2 - y1))
            
                    x1 = max(0, x1 + pad_x)
                    y1 = max(0, y1 + pad_y)
            
                    x2 = min(img.width, x2 - pad_x)
                    y2 = min(img.height, y2 - pad_y)
            
                    h_box = y2 - y1
                    w_box = x2 - x1
            
                    # full outfit crop
                    full_crop = img.crop((x1, y1, x2, y2))
            
                    # upper-body crop
                    upper_crop = img.crop((
                        x1,
                        y1,
                        x2,
                        int(y1 + 0.55 * h_box)
                    ))
            
                    # lower-body crop
                    lower_crop = img.crop((
                        x1,
                        int(y1 + 0.45 * h_box),
                        x2,
                        y2
                    ))
            
                    # -------------------------------------------------
                    # user selection
                    # -------------------------------------------------
            
                    if search_region == "Upper Body":
            
                        cropped = upper_crop
            
                    elif search_region == "Lower Body":
            
                        cropped = lower_crop
            
                    else:
            
                        cropped = full_crop
            
                    # -------------------------------------------------
                    # visualization
                    # -------------------------------------------------
            
                    st.markdown(
                        '<div class="section-label">DETECTED CROP</div>',
                        unsafe_allow_html=True
                    )
            
                    st.image(
                        cropped,
                        width=300
                    )

                    st.image(
                        cropped,
                        caption=f"YOLO crop (conf={conf:.2f})",
                        width="stretch"
                    )

                    st.success(f"Box: {[x1, y1, x2, y2]}")

                else:

                    cropped = img

                    st.warning(
                        "No confident detection — using full image"
                    )

            else:

                cropped = img

            st.session_state["final_crop"] = cropped

    with right:

        if uploaded and "final_crop" in st.session_state:

            final_crop = st.session_state["final_crop"]

            st.markdown(
                '<div class="section-label">RETRIEVAL RESULTS</div>',
                unsafe_allow_html=True
            )

            t0 = time.time()

            candidates = retrieve(
                final_crop,
                model,
                index,
                device,
                args.top_k
            )

            latency = (time.time() - t0) * 1000

            m1, m2, m3 = st.columns(3)

            m1.metric("Results", len(candidates))
            m2.metric("Index size", f"{len(index):,}")
            m3.metric("Latency", f"{latency:.0f}ms")

            st.markdown("---")

            dataset_root = Path(args.dataset_root)

            rows = [
                candidates[i:i+3]
                for i in range(0, len(candidates), 3)
            ]

            for row in rows:

                cols = st.columns(3)

                for col, cand in zip(cols, row):

                    with col:

                        img_path = (
                            dataset_root /
                            "img" /
                            cand["img_path"]
                        )

                        try:

                            result_img = Image.open(
                                img_path
                            ).convert("RGB")

                            st.image(
                                result_img,
                                width="stretch"
                            )

                        except Exception:

                            st.warning(
                                f"Missing image:\n{img_path}"
                            )

                        st.markdown(
                            f"""
                            **sim {cand['rerank_score']:.3f}**

                            {cand['item_id']}
                            """
                        )

                        if cand.get("caption"):

                            st.caption(
                                cand["caption"][:100]
                            )


if __name__ == "__main__":
    main()
