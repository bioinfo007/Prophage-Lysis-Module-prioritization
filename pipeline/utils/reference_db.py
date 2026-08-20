"""
pipeline/utils/reference_db.py
================================
Reference embedding database for ESM-2 similarity-based lysis module detection.

This is Stage 3 of the three-stage identification strategy:
  Stage 1: HMMER domain search (sequence-based, high precision)
  Stage 2: Keyword + topology (annotation-based)
  Stage 3: ESM-2 reference similarity ← this module
           (functional-space similarity, catches unannotated proteins)

The reference sets are built once using:
  python scripts/build_reference_embeddings.py --output references/

Then used in M02 for every pipeline run.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("reference_db")


class ReferenceEmbeddingDB:
    """
    Manages reference embedding matrices for all three lysis module tracks.

    Usage:
        db = ReferenceEmbeddingDB("references/")
        scores = db.score_protein("endolysin", query_embedding)
        is_candidate = scores["max_similarity"] > 0.70
    """

    # Default similarity thresholds — tuned conservatively
    # Lower = more sensitive but more false positives
    # Higher = more specific but misses divergent proteins
    # These are starting points — validate on your phages and adjust
    DEFAULT_THRESHOLDS = {
        "endolysin": 0.70,   # cosine similarity to nearest reference endolysin
        "holin":     0.72,   # holins are more conserved structurally → higher threshold
        "spanin":    0.68,   # spanins are very diverse → lower threshold
    }

    def __init__(
        self,
        ref_dir:    str,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.ref_dir    = Path(ref_dir)
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self._matrices: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, List[Dict]] = {}
        self._loaded:   Dict[str, bool]        = {}

        self._try_load_all()

    def _try_load_all(self) -> None:
        """Load all reference sets that exist. Skip missing ones gracefully."""
        for cls in ["endolysin", "holin", "spanin"]:
            npy_path  = self.ref_dir / f"{cls}_reference.npy"
            json_path = self.ref_dir / f"{cls}_reference.json"

            if npy_path.exists() and json_path.exists():
                try:
                    self._matrices[cls] = np.load(npy_path).astype(np.float32)
                    # L2-normalize reference matrix once at load time
                    norms = np.linalg.norm(self._matrices[cls], axis=1, keepdims=True)
                    norms = np.where(norms < 1e-10, 1e-10, norms)
                    self._matrices[cls] /= norms

                    self._metadata[cls] = json.loads(json_path.read_text())
                    self._loaded[cls]   = True
                    log.info(
                        f"Reference set loaded: {cls} "
                        f"({self._matrices[cls].shape[0]} sequences)"
                    )
                except Exception as e:
                    log.warning(f"Could not load {cls} reference set: {e}")
                    self._loaded[cls] = False
            else:
                self._loaded[cls] = False
                log.debug(f"Reference set not found: {cls} ({npy_path})")

    def is_available(self, class_name: str) -> bool:
        return self._loaded.get(class_name, False)

    def any_available(self) -> bool:
        return any(self._loaded.values())

    def score_protein(
        self,
        class_name: str,
        embedding:  np.ndarray,
        top_k:      int = 5,
    ) -> Dict:
        """
        Score a query protein embedding against the reference set.

        Args:
            class_name: "endolysin" | "holin" | "spanin"
            embedding:  (1280,) float32 — must be from same ESM-2 model
            top_k:      number of nearest references to return

        Returns:
            dict with:
                max_similarity:  float — cosine similarity to nearest reference
                mean_top_k:      float — mean similarity of top-k nearest
                is_candidate:    bool  — True if max_similarity > threshold
                nearest:         list of (uniprot_id, name, similarity) tuples
                threshold_used:  float
        """
        if not self.is_available(class_name):
            return {
                "max_similarity": 0.0,
                "mean_top_k":     0.0,
                "is_candidate":   False,
                "nearest":        [],
                "threshold_used": self.thresholds[class_name],
                "reference_available": False,
            }

        ref_matrix = self._matrices[class_name]
        metadata   = self._metadata[class_name]
        threshold  = self.thresholds[class_name]

        # L2-normalize query
        q = embedding.astype(np.float32).flatten()
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return {
                "max_similarity": 0.0,
                "mean_top_k":     0.0,
                "is_candidate":   False,
                "nearest":        [],
                "threshold_used": threshold,
                "reference_available": True,
            }
        q = q / norm

        # Cosine similarity to all references (matrix is already normalized)
        similarities = ref_matrix @ q   # (N_ref,)
        similarities = np.clip(similarities, -1.0, 1.0)

        # Top-k nearest
        top_k_actual = min(top_k, len(similarities))
        top_indices  = np.argsort(similarities)[::-1][:top_k_actual]

        nearest = []
        for idx in top_indices:
            meta = metadata[idx] if idx < len(metadata) else {}
            nearest.append({
                "uniprot_id": meta.get("uniprot_id", f"ref_{idx}"),
                "name":       meta.get("name", "unknown"),
                "organism":   meta.get("organism", "unknown"),
                "similarity": float(similarities[idx]),
            })

        max_sim    = float(similarities[top_indices[0]]) if top_indices.size > 0 else 0.0
        mean_top_k = float(np.mean(similarities[top_indices])) if top_indices.size > 0 else 0.0

        return {
            "max_similarity":       max_sim,
            "mean_top_k":           mean_top_k,
            "is_candidate":         max_sim >= threshold,
            "nearest":              nearest,
            "threshold_used":       threshold,
            "reference_available":  True,
        }

    def batch_score(
        self,
        class_name:  str,
        embeddings:  np.ndarray,     # (N, 1280)
        protein_ids: List[str],
        top_k:       int = 3,
    ) -> List[Dict]:
        """
        Score a batch of query embeddings efficiently using matrix multiplication.
        Much faster than calling score_protein() in a loop for large batches.

        Returns list of score dicts, one per protein.
        """
        if not self.is_available(class_name):
            return [
                {"max_similarity": 0.0, "is_candidate": False,
                 "reference_available": False}
                for _ in protein_ids
            ]

        ref_matrix = self._matrices[class_name]
        metadata   = self._metadata[class_name]
        threshold  = self.thresholds[class_name]

        # Normalize query matrix
        Q     = embeddings.astype(np.float32)
        norms = np.linalg.norm(Q, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        Q     = Q / norms

        # All-vs-all cosine similarity: (N_query, N_ref)
        sim_matrix = Q @ ref_matrix.T
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

        results = []
        for i, pid in enumerate(protein_ids):
            sims       = sim_matrix[i]
            top_k_act  = min(top_k, len(sims))
            top_idx    = np.argsort(sims)[::-1][:top_k_act]
            max_sim    = float(sims[top_idx[0]]) if top_idx.size > 0 else 0.0
            mean_top_k = float(np.mean(sims[top_idx])) if top_idx.size > 0 else 0.0

            nearest = []
            for idx in top_idx:
                meta = metadata[idx] if idx < len(metadata) else {}
                nearest.append({
                    "uniprot_id": meta.get("uniprot_id", f"ref_{idx}"),
                    "name":       meta.get("name", "unknown"),
                    "similarity": float(sims[idx]),
                })

            results.append({
                "protein_id":           pid,
                "max_similarity":       max_sim,
                "mean_top_k":           mean_top_k,
                "is_candidate":         max_sim >= threshold,
                "nearest":              nearest,
                "threshold_used":       threshold,
                "reference_available":  True,
            })

        return results

    def summary(self) -> str:
        lines = ["Reference DB summary:"]
        for cls in ["endolysin", "holin", "spanin"]:
            if self._loaded.get(cls):
                n = self._matrices[cls].shape[0]
                t = self.thresholds[cls]
                lines.append(f"  {cls}: {n} references, threshold={t:.2f}")
            else:
                lines.append(f"  {cls}: NOT LOADED")
        return "\n".join(lines)
