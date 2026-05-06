"""
HNSW-based ANN vector index for fast product retrieval.

Uses FAISS HNSW (Hierarchical Navigable Small World) for:
  - O(log N) approximate nearest-neighbour search
  - Low memory footprint
  - No GPU required at query time

Stores alongside the index:
  - item_ids   : list of item_id strings for each vector
  - captions   : list of generated captions for ITM re-ranking
  - img_paths  : list of relative image paths for display
  - metadata   : arbitrary per-item JSON metadata
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import faiss
import torch


# ─────────────────────────────────────────────
#  Index
# ─────────────────────────────────────────────

class HNSWIndex:
    """
    FAISS HNSW index wrapping gallery embeddings.

    Args:
        dim          : embedding dimensionality
        M            : HNSW graph connectivity (default 32; higher = better recall, more RAM)
        ef_construction : HNSW build-time search depth (higher = better quality index)
        ef_search    : HNSW query-time search depth (higher = better recall, slower)
    """

    def __init__(
        self,
        dim: int = 256,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ):
        self.dim = dim
        # Inner-product on L2-normalised vectors ≡ cosine similarity
        self._index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = ef_construction
        self._index.hnsw.efSearch = ef_search

        self.item_ids : list[str] = []
        self.captions : list[str] = []
        self.img_paths: list[str] = []
        self.metadata : list[dict] = []

    # ── Build ─────────────────────────────────

    def add(
        self,
        embeddings: np.ndarray,
        item_ids  : list[str],
        img_paths : list[str],
        captions  : list[str] | None = None,
        metadata  : list[dict] | None = None,
    ):
        """
        Add vectors and associated metadata to the index.

        Args:
            embeddings : (N, dim) float32 L2-normalised vectors
            item_ids   : length-N list of item_id strings
            img_paths  : length-N list of relative image paths
            captions   : length-N list of BLIP-2 captions (None → empty strings)
            metadata   : length-N list of extra dicts (None → empty dicts)
        """
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.dim
        vecs = embeddings.astype(np.float32)
        # Re-normalise in case of floating-point drift
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-8, None)

        self._index.add(vecs)
        self.item_ids.extend(item_ids)
        self.img_paths.extend(img_paths)
        self.captions.extend(captions or [""] * len(item_ids))
        self.metadata.extend(metadata or [{} for _ in item_ids])

    def __len__(self) -> int:
        return self._index.ntotal

    # ── Query ─────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
    ) -> list[dict]:
        """
        Search for the top-K nearest gallery items.

        Args:
            query_embedding : (D,) or (1, D) float32 L2-normalised vector
        Returns:
            List of dicts sorted by descending cosine similarity:
            [{'rank', 'score', 'item_id', 'img_path', 'caption', 'metadata'}, ...]
        """
        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        q /= np.linalg.norm(q) + 1e-8

        scores, indices = self._index.search(q, top_k)
        scores  = scores[0].tolist()
        indices = indices[0].tolist()

        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0:   # FAISS returns -1 for unfilled slots
                continue
            results.append({
                "rank"    : rank + 1,
                "score"   : float(score),
                "item_id" : self.item_ids[idx],
                "img_path": self.img_paths[idx],
                "caption" : self.captions[idx],
                "metadata": self.metadata[idx],
            })
        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 50,
    ) -> list[list[dict]]:
        """Search multiple queries at once. Returns list-of-lists."""
        Q = query_embeddings.astype(np.float32)
        norms = np.linalg.norm(Q, axis=1, keepdims=True)
        Q /= np.clip(norms, 1e-8, None)

        scores_mat, indices_mat = self._index.search(Q, top_k)
        all_results = []
        for scores, indices in zip(scores_mat, indices_mat):
            results = []
            for rank, (score, idx) in enumerate(zip(scores, indices)):
                if idx < 0:
                    continue
                results.append({
                    "rank"    : rank + 1,
                    "score"   : float(score),
                    "item_id" : self.item_ids[idx],
                    "img_path": self.img_paths[idx],
                    "caption" : self.captions[idx],
                    "metadata": self.metadata[idx],
                })
            all_results.append(results)
        return all_results

    # ── Persistence ───────────────────────────

    def save(self, save_dir: str):
        """Save FAISS index + metadata to directory."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(Path(save_dir) / "hnsw.index"))
        meta = {
            "dim"      : self.dim,
            "item_ids" : self.item_ids,
            "captions" : self.captions,
            "img_paths": self.img_paths,
            "metadata" : self.metadata,
        }
        with open(Path(save_dir) / "metadata.json", "w") as f:
            json.dump(meta, f)
        print(f"[Index] Saved {len(self)} vectors → {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> "HNSWIndex":
        """Load a previously saved index from directory."""
        index_path = Path(save_dir) / "hnsw.index"
        meta_path  = Path(save_dir) / "metadata.json"

        faiss_idx = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            meta = json.load(f)

        obj = cls.__new__(cls)
        obj.dim        = meta["dim"]
        obj._index     = faiss_idx
        obj.item_ids   = meta["item_ids"]
        obj.captions   = meta["captions"]
        obj.img_paths  = meta["img_paths"]
        obj.metadata   = meta["metadata"]
        print(f"[Index] Loaded {len(obj)} vectors from {save_dir}")
        return obj
