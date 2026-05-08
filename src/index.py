"""
Improved HNSW ANN retrieval index.

Enhancements:
- category-aware metadata
- duplicate suppression
- reranking support
- robust metadata handling
- normalized embeddings
- better retrieval stability
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import faiss


# =========================================================
# INDEX
# =========================================================

class HNSWIndex:

    def __init__(
        self,
        dim: int = 256,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 256,
    ):

        self.dim = dim

        self._index = faiss.IndexHNSWFlat(
            dim,
            M,
            faiss.METRIC_INNER_PRODUCT
        )

        self._index.hnsw.efConstruction = ef_construction
        self._index.hnsw.efSearch = ef_search

        self.item_ids = []
        self.captions = []
        self.img_paths = []
        self.metadata = []

    # =====================================================
    # ADD
    # =====================================================

    def add(
        self,
        embeddings: np.ndarray,
        item_ids: list[str],
        img_paths: list[str],
        captions: list[str] | None = None,
        metadata: list[dict] | None = None,
    ):

        assert embeddings.ndim == 2
        assert embeddings.shape[1] == self.dim

        vecs = embeddings.astype(np.float32)

        # normalize

        norms = np.linalg.norm(
            vecs,
            axis=1,
            keepdims=True
        )

        vecs = vecs / np.clip(
            norms,
            1e-8,
            None
        )

        self._index.add(vecs)

        captions = captions or [""] * len(item_ids)
        metadata = metadata or [{} for _ in item_ids]

        # ---------------------------------------------
        # enrich metadata
        # ---------------------------------------------

        enriched_meta = []

        for path, meta in zip(img_paths, metadata):

            path_lower = str(path).lower()

            category = "unknown"

            if any(k in path_lower for k in [
                "tee",
                "shirt",
                "top",
                "blouse",
                "tank"
            ]):
                category = "top"

            elif any(k in path_lower for k in [
                "pant",
                "short",
                "skirt",
                "legging",
                "jean"
            ]):
                category = "bottom"

            elif "dress" in path_lower:
                category = "dress"

            gender = "unknown"

            if "women" in path_lower:
                gender = "women"

            elif "men" in path_lower:
                gender = "men"

            new_meta = {
                **meta,
                "category": category,
                "gender": gender,
            }

            enriched_meta.append(new_meta)

        self.item_ids.extend(item_ids)
        self.img_paths.extend(img_paths)
        self.captions.extend(captions)
        self.metadata.extend(enriched_meta)

    # =====================================================
    # LEN
    # =====================================================

    def __len__(self):

        return self._index.ntotal

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        query_category: str | None = None,
        query_gender: str | None = None,
        deduplicate_items: bool = True,
    ) -> list[dict]:

        q = np.array(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        q /= np.linalg.norm(q) + 1e-8

        scores, indices = self._index.search(
            q,
            max(top_k * 3, 50)
        )

        scores = scores[0].tolist()
        indices = indices[0].tolist()

        results = []

        seen_items = set()

        for rank, (score, idx) in enumerate(
            zip(scores, indices)
        ):

            if idx < 0:
                continue

            item_id = self.item_ids[idx]

            # -----------------------------------------
            # duplicate suppression
            # -----------------------------------------

            if deduplicate_items:

                if item_id in seen_items:
                    continue

                seen_items.add(item_id)

            meta = self.metadata[idx]

            # -----------------------------------------
            # reranking bonuses
            # -----------------------------------------

            rerank_score = float(score)

            if query_category is not None:

                if meta.get("category") == query_category:
                    rerank_score += 0.08

            if query_gender is not None:

                if meta.get("gender") == query_gender:
                    rerank_score += 0.04

            # -----------------------------------------
            # texture/style heuristic
            # -----------------------------------------

            caption = self.captions[idx]

            if caption:

                caption_lower = caption.lower()

                if any(k in caption_lower for k in [
                    "oversized",
                    "casual",
                    "loose",
                    "streetwear"
                ]):
                    rerank_score += 0.02

            results.append({
                "rank": rank + 1,
                "score": float(score),
                "rerank_score": float(rerank_score),
                "item_id": item_id,
                "img_path": self.img_paths[idx],
                "caption": caption,
                "metadata": meta,
            })

        # ---------------------------------------------
        # final reranking
        # ---------------------------------------------

        results = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return results[:top_k]

    # =====================================================
    # BATCH SEARCH
    # =====================================================

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 50,
    ):

        outputs = []

        for q in query_embeddings:

            outputs.append(
                self.search(
                    q,
                    top_k=top_k
                )
            )

        return outputs

    # =====================================================
    # SAVE
    # =====================================================

    def save(self, save_dir: str):

        save_dir = Path(save_dir)

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self._index,
            str(save_dir / "hnsw.index")
        )

        meta = {
            "dim": self.dim,
            "item_ids": self.item_ids,
            "captions": self.captions,
            "img_paths": self.img_paths,
            "metadata": self.metadata,
        }

        with open(
            save_dir / "metadata.json",
            "w"
        ) as f:

            json.dump(meta, f)

        print(
            f"[Index] Saved {len(self)} vectors → {save_dir}"
        )

    # =====================================================
    # LOAD
    # =====================================================

    @classmethod
    def load(
        cls,
        save_dir: str
    ) -> "HNSWIndex":

        save_dir = Path(save_dir)

        index_path = save_dir / "hnsw.index"
        meta_path = save_dir / "metadata.json"

        faiss_idx = faiss.read_index(
            str(index_path)
        )

        with open(meta_path) as f:

            meta = json.load(f)

        obj = cls.__new__(cls)

        obj.dim = meta["dim"]
        obj._index = faiss_idx
        obj.item_ids = meta["item_ids"]
        obj.captions = meta["captions"]
        obj.img_paths = meta["img_paths"]
        obj.metadata = meta["metadata"]

        print(
            f"[Index] Loaded {len(obj)} vectors from {save_dir}"
        )

        return obj
