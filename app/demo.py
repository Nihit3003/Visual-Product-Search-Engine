"""
Visual Product Search Engine — Streamlit Demo
=============================================
Interactive web app demonstrating end-to-end retrieval:
  1. Upload a query image
  2. YOLO detects & crops the main clothing item
  3. User confirms crop (or re-crops manually)
  4. CLIP encodes the query
  5. HNSW index returns top-K candidates
  6. BLIP-2 ITM re-ranks candidates
  7. Results displayed with metadata and similarity scores

Run:
    streamlit run app/demo.py -- \
        --dataset_root /kaggle/input/deepfashion-inshop \
        --index_dir    /kaggle/working/index/condition_C_alpha0.6 \
        --ckpt_path    /kaggle/working/checkpoints/clip_finetuned_best.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model     import VisualSearchModel
from src.index     import HNSWIndex
from src.localizer import YOLOLocalizer
from src.blip_module import ITMReranker
from src.dataset   import get_clip_transform


# ──────────────────────────────────────────────────
#  CLI args (passed after `--` to streamlit)
# ──────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="/kaggle/input/deepfashion-inshop")
    p.add_argument("--index_dir",    default="/kaggle/working/index/condition_C_alpha0.6")
    p.add_argument("--ckpt_path",    default=None)
    p.add_argument("--embed_dim",    type=int,   default=256)
    p.add_argument("--alpha",        type=float, default=0.6)
    p.add_argument("--top_k",        type=int,   default=15)
    p.add_argument("--yolo_weights", default="yolov8n.pt")
    p.add_argument("--use_itm",      action="store_true", default=True)
    return p.parse_args()


# ──────────────────────────────────────────────────
#  Cached resource loaders
# ──────────────────────────────────────────────────

@st.cache_resource
def load_model(ckpt_path, alpha, embed_dim):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = VisualSearchModel(alpha=alpha, embed_dim=embed_dim, unfreeze_last_n=4)
    if ckpt_path and Path(ckpt_path).exists():
        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
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


@st.cache_resource
def load_reranker(use_itm):
    if not use_itm:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return ITMReranker(device=device)
    except Exception:
        return None


# ──────────────────────────────────────────────────
#  Retrieval helper
# ──────────────────────────────────────────────────

def retrieve(query_img: Image.Image, model, index, reranker, device, top_k, args):
    transform = get_clip_transform(224, augment=False)
    tensor    = transform(query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model.encode_image(tensor).cpu().numpy()[0]

    candidates = index.search(q_emb, top_k=min(top_k * 3, 50))

    if reranker is not None and candidates:
        candidates = reranker.rerank(query_img, candidates)

    return candidates[:top_k]


# ──────────────────────────────────────────────────
#  Page layout & styling
# ──────────────────────────────────────────────────

def apply_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #0f0f0f 0%, #555 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #777;
        margin-bottom: 2rem;
    }
    .result-card {
        border: 1.5px solid #e8e8e8;
        border-radius: 12px;
        padding: 10px;
        background: #fafafa;
        transition: all 0.2s;
        margin-bottom: 12px;
    }
    .result-card:hover {
        border-color: #222;
        box-shadow: 4px 4px 0px #222;
        transform: translate(-2px, -2px);
    }
    .score-badge {
        display: inline-block;
        background: #111;
        color: #fff;
        font-size: 0.7rem;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    .rank-badge {
        display: inline-block;
        background: #f3f3f3;
        color: #555;
        font-size: 0.65rem;
        font-family: 'DM Sans', sans-serif;
        padding: 2px 7px;
        border-radius: 20px;
        margin-left: 4px;
    }
    .section-label {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #aaa;
        margin-bottom: 8px;
    }
    .crop-confirm-btn button {
        background: #111 !important;
        color: #fff !important;
        font-family: 'Syne', sans-serif !important;
        border-radius: 8px !important;
    }
    .stButton > button {
        font-family: 'Syne', sans-serif;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background: #f5f5f5;
        border-radius: 10px;
        padding: 10px 14px;
    }
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────
#  Main app
# ──────────────────────────────────────────────────

def main():
    args = parse_args()

    st.set_page_config(
        page_title="Visual Product Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_style()

    # ── Header ────────────────────────────────
    st.markdown('<div class="hero-title">Visual Product Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a clothing image → find visually similar products instantly</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Load resources ─────────────────────
    with st.spinner("Loading models and index…"):
        model, device   = load_model(args.ckpt_path, args.alpha, args.embed_dim)
        index           = load_index(args.index_dir)
        localizer       = load_localizer(args.yolo_weights)
        reranker        = load_reranker(args.use_itm)

    # Sidebar info
    with st.sidebar:
        st.markdown("### ⚙️ Config")
        st.write(f"**Index size:** {len(index):,}")
        st.write(f"**Device:** {device.upper()}")
        st.write(f"**Alpha:** {args.alpha}")
        st.write(f"**YOLO:** {'✓' if localizer else '✗ (GT bbox)'}")
        st.write(f"**ITM Re-rank:** {'✓' if reranker else '✗'}")
        st.markdown("---")
        top_k = st.slider("Top-K results", 5, 20, args.top_k)
        st.markdown("### 📖 About")
        st.caption("DeepFashion In-Shop • CLIP + BLIP-2 + YOLO • HNSW ANN")

    # ── Upload ────────────────────────────
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown('<div class="section-label">Query Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

        if uploaded is not None:
            query_img = Image.open(uploaded).convert("RGB")
            st.image(query_img, caption="Original", use_column_width=True)

            # ── YOLO detection ────────────────
            st.markdown('<div class="section-label" style="margin-top:16px">Detected Crop</div>',
                        unsafe_allow_html=True)
            if localizer is not None:
                detection = localizer.detect(query_img)
                cropped   = detection["cropped"]
                conf      = detection["confidence"]
                box       = detection["box"]

                if box:
                    st.image(cropped, caption=f"YOLO crop (conf={conf:.2f})", use_column_width=True)
                    st.success(f"Box: {box}")
                else:
                    st.image(cropped, caption="No detection — using full image", use_column_width=True)
                    st.warning("No confident detection. Using full image.")
            else:
                cropped = query_img
                st.image(cropped, caption="GT crop / full image", use_column_width=True)

            # ── Crop confirmation ─────────────
            st.markdown('<div class="section-label" style="margin-top:16px">Confirm</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            confirm_crop = c1.button("✅ Confirm Crop", use_container_width=True)
            use_full     = c2.button("🔄 Use Full Image", use_container_width=True)

            if "final_crop" not in st.session_state:
                st.session_state.final_crop = None

            if confirm_crop:
                st.session_state.final_crop = cropped
            if use_full:
                st.session_state.final_crop = query_img

    # ── Results ───────────────────────────
    with col2:
        if uploaded is not None and st.session_state.get("final_crop") is not None:
            final_crop = st.session_state.final_crop

            st.markdown('<div class="section-label">Retrieval Results</div>', unsafe_allow_html=True)

            with st.spinner("Searching…"):
                t0         = __import__("time").time()
                candidates = retrieve(final_crop, model, index, reranker, device, top_k, args)
                elapsed    = __import__("time").time() - t0

            # Metrics bar
            m1, m2, m3 = st.columns(3)
            m1.metric("Results", len(candidates))
            m2.metric("Index size", f"{len(index):,}")
            m3.metric("Latency", f"{elapsed*1000:.0f}ms")
            st.markdown("---")

            # Display results in a grid
            n_cols = 3
            rows   = [candidates[i : i + n_cols] for i in range(0, len(candidates), n_cols)]
            dataset_root = Path(args.dataset_root)

            for row in rows:
                cols = st.columns(n_cols)
                for col, cand in zip(cols, row):
                    with col:
                        img_path = dataset_root / "img" / cand["img_path"]
                        try:
                            img = Image.open(str(img_path)).convert("RGB")
                            st.image(img, use_column_width=True)
                        except Exception:
                            st.image(
                                Image.new("RGB", (150, 200), color=(230, 230, 230)),
                                use_column_width=True,
                            )
                        score_str = f"{cand['score']:.3f}"
                        itm_str   = f"ITM {cand.get('itm_score', 0):.2f}" \
                                    if "itm_score" in cand else ""
                        st.markdown(
                            f'<span class="score-badge">sim {score_str}</span>'
                            + (f'<span class="rank-badge">{itm_str}</span>' if itm_str else "")
                            + f'<br><small style="color:#888">{cand["item_id"]}</small>',
                            unsafe_allow_html=True,
                        )
                        if cand.get("caption"):
                            st.caption(cand["caption"][:80] + "…" if len(cand["caption"]) > 80
                                       else cand["caption"])

        elif uploaded is not None:
            st.info("👆 Confirm or reject the crop to start searching.")
        else:
            st.markdown("""
            <div style="border: 2px dashed #ddd; border-radius: 16px; padding: 3rem;
                        text-align: center; color: #aaa; margin-top: 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">👗</div>
                <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;">
                    Upload a clothing image to begin
                </div>
                <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                    Supports JPG, PNG, WebP
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
