"""
Improved HNSW ANN retrieval index.
"""

from __future__ import annotations

import json
from pathlib import Path

import hnswlib
import numpy as np


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

        self.M = M

        self.ef_construction = ef_construction

        self.ef_search = ef_search

        self._index = None

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

        vecs = embeddings.astype(
            np.float32
        )

        # -------------------------------------------------
        # normalize
        # -------------------------------------------------

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

        n = len(vecs)

        # -------------------------------------------------
        # lazy init
        # -------------------------------------------------

        if self._index is None:

            self._index = hnswlib.Index(
                space="cosine",
                dim=self.dim
            )

            self._index.init_index(
                max_elements=50000,
                ef_construction=self.ef_construction,
                M=self.M
            )

            self._index.set_ef(
                self.ef_search
            )

        current_count = len(
            self.item_ids
        )

        labels = np.arange(
            current_count,
            current_count + n
        )

        self._index.add_items(
            vecs,
            labels
        )

        captions = (
            captions
            or
            [""] * len(item_ids)
        )

        metadata = (
            metadata
            or
            [{} for _ in item_ids]
        )

        enriched_meta = []

        # -------------------------------------------------
        # enrich metadata
        # -------------------------------------------------

        for path, meta in zip(
            img_paths,
            metadata
        ):

            path_lower = str(
                path
            ).lower()

            category = "unknown"

            if any(k in path_lower for k in [
                "tee",
                "shirt",
                "top",
                "blouse",
                "tank",
                "hoodie",
                "jacket",
            ]):
                category = "top"

            elif any(k in path_lower for k in [
                "pant",
                "short",
                "skirt",
                "legging",
                "jean",
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

                "category":
                    category,

                "gender":
                    gender,
            }

            enriched_meta.append(
                new_meta
            )

        self.item_ids.extend(
            item_ids
        )

        self.img_paths.extend(
            img_paths
        )

        self.captions.extend(
            captions
        )

        self.metadata.extend(
            enriched_meta
        )

    # =====================================================
    # LEN
    # =====================================================

    def __len__(self):

        return len(self.item_ids)

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        query_category: str | None = None,
        query_gender: str | None = None,
        query_region: str | None = None,
        deduplicate_items: bool = True,
    ) -> list[dict]:

        q = np.array(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        q /= (
            np.linalg.norm(q)
            + 1e-8
        )

        indices, distances = self._index.knn_query(
            q,
            k=max(top_k * 4, 80)
        )

        indices = indices[0].tolist()

        distances = distances[0].tolist()

        results = []

        seen_items = set()

        # -------------------------------------------------
        # candidate loop
        # -------------------------------------------------

        for rank, (idx, dist) in enumerate(
            zip(indices, distances)
        ):

            if idx < 0:
                continue

            item_id = self.item_ids[idx]

            # ---------------------------------------------
            # deduplicate
            # ---------------------------------------------

            if deduplicate_items:

                if item_id in seen_items:
                    continue

                seen_items.add(item_id)

            meta = self.metadata[idx]

            # ---------------------------------------------
            # region-aware filtering
            # ---------------------------------------------

            if query_region == "upper":

                if meta.get("category") not in [
                    "top",
                    "shirt",
                    "upper",
                    "hoodie",
                    "jacket",
                    "blouse",
                ]:
                    continue

            elif query_region == "lower":

                if meta.get("category") not in [
                    "bottom",
                    "pants",
                    "shorts",
                    "skirt",
                    "jeans",
                    "legging",
                ]:
                    continue

            similarity = 1.0 - float(dist)

            rerank_score = similarity

            # ---------------------------------------------
            # metadata reranking
            # ---------------------------------------------

            if (
                query_category is not None
                and
                meta.get("category")
                == query_category
            ):

                rerank_score += 0.15

            if (
                query_gender is not None
                and
                meta.get("gender")
                == query_gender
            ):

                rerank_score += 0.04

            # ---------------------------------------------
            # caption reranking
            # ---------------------------------------------

            caption = self.captions[idx]

            if caption:

                caption_lower = caption.lower()

                if any(k in caption_lower for k in [

                    "oversized",
                    "casual",
                    "streetwear",
                    "loose",
                    "minimalist",
                    "athletic",
                ]):

                    rerank_score += 0.02

            results.append({

                "rank":
                    rank + 1,

                "score":
                    similarity,

                "rerank_score":
                    rerank_score,

                "item_id":
                    item_id,

                "img_path":
                    self.img_paths[idx],

                "caption":
                    caption,

                "metadata":
                    meta,
            })

        # -------------------------------------------------
        # final rerank
        # -------------------------------------------------

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

    def save(
        self,
        save_dir: str
    ):

        save_dir = Path(save_dir)

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self._index.save_index(
            str(
                save_dir /
                "hnsw.bin"
            )
        )

        meta = {

            "dim":
                self.dim,

            "item_ids":
                self.item_ids,

            "captions":
                self.captions,

            "img_paths":
                self.img_paths,

            "metadata":
                self.metadata,
        }

        with open(
            save_dir / "metadata.json",
            "w"
        ) as f:

            json.dump(
                meta,
                f
            )

        print(
            f"[Index] Saved "
            f"{len(self)} vectors"
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

        with open(
            save_dir / "metadata.json"
        ) as f:

            meta = json.load(f)

        obj = cls(
            dim=meta["dim"]
        )

        obj.item_ids = meta["item_ids"]

        obj.captions = meta["captions"]

        obj.img_paths = meta["img_paths"]

        obj.metadata = meta["metadata"]

        obj._index = hnswlib.Index(
            space="cosine",
            dim=obj.dim
        )

        obj._index.load_index(
            str(
                save_dir /
                "hnsw.bin"
            )
        )

        obj._index.set_ef(
            obj.ef_search
        )

        print(
            f"[Index] Loaded "
            f"{len(obj)} vectors"
        )

        return obj
