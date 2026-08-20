"""
m05_clustering.py
=================
Module 05: Per-track UMAP + HDBSCAN clustering.

Runs three independent clustering analyses — one per track.
Cross-track clustering is avoided: a holin and endolysin should not share
a cluster even if their ESM-2 embeddings happen to be close.

UMAP fitted once in 50D → HDBSCAN on 50D → 2D derived from 50D coordinates.
Noise points are assigned to nearest cluster centroid rather than left as orphans.

Input:  data/intermediate/03_gate1/candidates_passing.json
        data/intermediate/04_embeddings/embedding_matrix.npy
        data/intermediate/04_embeddings/embedding_index.json
Output: data/intermediate/05_clusters/{track}_cluster_assignments.tsv
        data/intermediate/05_clusters/{track}_umap_2d.tsv
        data/intermediate/05_clusters/cluster_summary.json
        data/intermediate/03_gate1/candidates_passing.json (updated)
"""

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import umap
import hdbscan

from pipeline.utils.data_model import (
    _BaseRecord, load_candidates, save_candidates, split_by_track,
)

log = logging.getLogger("m05_clustering")


def run(cfg: dict) -> None:
    paths  = cfg["paths"]
    cl_cfg = cfg["clustering"]

    in_dir  = Path(paths["intermediate_dir"]) / "03_gate1"
    emb_dir = Path(paths["intermediate_dir"]) / "04_embeddings"
    out_dir = Path(paths["intermediate_dir"]) / "05_clusters"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path  = in_dir / "candidates_passing.json"
    candidates = load_candidates(str(cand_path))
    cand_by_id = {c.candidate_id: c for c in candidates}

    # Load full embedding matrix and index
    matrix_path = emb_dir / "embedding_matrix.npy"
    index_path  = emb_dir / "embedding_index.json"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Embedding matrix not found: {matrix_path}")

    full_matrix = np.load(matrix_path)
    full_index  = json.loads(index_path.read_text())
    id_to_row   = {cid: i for i, cid in enumerate(full_index)}

    tracks = split_by_track(candidates)
    all_summaries = []

    for track_name, track_cands in tracks.items():
        if not track_cands:
            log.info(f"  Track '{track_name}': empty — skipping")
            continue

        log.info(f"  Clustering track '{track_name}': {len(track_cands)} candidates")

        # Extract sub-matrix for this track
        track_ids  = [c.candidate_id for c in track_cands if c.candidate_id in id_to_row]
        track_rows = [full_matrix[id_to_row[cid]] for cid in track_ids]

        # UMAP requires at least 4 samples to function reliably
        # (spectral init needs k < N, and N must be > n_components)
        if len(track_ids) < 4:
            log.warning(
                f"    Too few candidates ({len(track_ids)}) — "
                f"skipping UMAP, assigning all to cluster 0"
            )
            for i, cid in enumerate(track_ids):
                c = cand_by_id.get(cid)
                if c is None:
                    continue
                c.cluster_id         = 0
                c.is_noise           = False
                c.cluster_enrichment = "singleton"
                c.umap_x             = float(i)
                c.umap_y             = 0.0
            _write_cluster_assignments(
                out_dir / f"{track_name}_cluster_assignments.tsv",
                track_ids, [0] * len(track_ids), {0: "singleton"},
            )
            _write_umap_2d(
                out_dir / f"{track_name}_umap_2d.tsv",
                track_ids,
                np.array([[float(i), 0.0] for i in range(len(track_ids))]),
                [0] * len(track_ids),
            )
            all_summaries.append({
                "track": track_name, "n_clusters": 1, "n_noise": 0,
                "n_candidates": len(track_ids), "cluster_enrichments": {0: "singleton"},
            })
            continue

        matrix = np.vstack(track_rows).astype(np.float32)

        labels_50d, embedding_2d = _cluster_track(matrix, cl_cfg, track_name)

        # Assign noise points to nearest cluster
        labels_final = _assign_noise_to_nearest(labels_50d, matrix)

        n_clusters = len(set(labels_final)) - (1 if -1 in labels_final else 0)
        n_noise    = int((np.array(labels_final) == -1).sum())

        log.info(
            f"    {track_name}: {n_clusters} clusters | "
            f"{n_noise} remaining noise points"
        )

        # Cluster enrichment annotation
        enrichment = _annotate_enrichment(
            labels_final, track_ids,
            {cid: cand_by_id[cid] for cid in track_ids if cid in cand_by_id},
        )

        # Update candidate records
        for i, cid in enumerate(track_ids):
            c = cand_by_id.get(cid)
            if c is None:
                continue
            c.cluster_id         = int(labels_final[i])
            c.is_noise           = bool(labels_final[i] == -1)
            c.umap_x             = float(embedding_2d[i, 0])
            c.umap_y             = float(embedding_2d[i, 1])
            c.cluster_enrichment = enrichment.get(int(labels_final[i]), "unknown")

        # Write per-track outputs
        _write_cluster_assignments(
            out_dir / f"{track_name}_cluster_assignments.tsv",
            track_ids, labels_final, enrichment,
        )
        _write_umap_2d(
            out_dir / f"{track_name}_umap_2d.tsv",
            track_ids, embedding_2d, labels_final,
        )

        all_summaries.append({
            "track":        track_name,
            "n_clusters":   n_clusters,
            "n_noise":      n_noise,
            "n_candidates": len(track_ids),
            "cluster_enrichments": dict(enrichment),
        })

    # Save updated candidates
    save_candidates(candidates, str(cand_path))

    # Write cluster summary
    (out_dir / "cluster_summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )

    log.info(f"M05 complete — clustering done for {len(all_summaries)} tracks")


# ── Core clustering logic ─────────────────────────────────────────────────────

def _cluster_track(
    matrix: np.ndarray,
    cl_cfg: dict,
    track:  str,
) -> Tuple[List[int], np.ndarray]:
    """
    UMAP (50D) → HDBSCAN → UMAP (2D from 50D init).
    Returns (cluster_labels, embedding_2d).
    """
    n = matrix.shape[0]

    # Clamp n_neighbors to valid range (must be < n)
    n_neighbors = min(cl_cfg.get("umap_n_neighbors", 15), n - 1)
    # n_components must be < n_samples for UMAP spectral init
    # Use n-2 as safe upper bound (eigsh requires k < N)
    n_components_50 = min(cl_cfg.get("umap_n_components", 50), n - 2, 50)
    n_components_50 = max(n_components_50, 2)  # must be at least 2

    log.debug(f"    UMAP 50D: n_neighbors={n_neighbors}, n_components={n_components_50}")

    # UMAP 50D for clustering
    reducer_50 = umap.UMAP(
        n_components = n_components_50,
        n_neighbors  = n_neighbors,
        min_dist     = cl_cfg.get("umap_min_dist", 0.1),
        metric       = cl_cfg.get("umap_metric", "cosine"),
        random_state = cl_cfg.get("umap_random_state", 42),
        low_memory   = True,
        verbose      = False,
    )
    embedding_50 = reducer_50.fit_transform(matrix)

    # HDBSCAN on 50D
    min_cluster = max(2, min(
        cl_cfg.get("hdbscan_min_cluster_size", 3),
        max(2, n // 10),   # auto-scale for very small datasets
    ))
    min_samples = max(1, min(
        cl_cfg.get("hdbscan_min_samples", 2),
        min_cluster - 1,
    ))

    log.debug(f"    HDBSCAN: min_cluster_size={min_cluster}, min_samples={min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster,
        min_samples      = min_samples,
        metric           = cl_cfg.get("hdbscan_metric", "euclidean"),
        core_dist_n_jobs = -1,   # parallel distance computation
    )
    labels_50d = clusterer.fit_predict(embedding_50).tolist()

    # UMAP 2D for visualization — initialized from 50D spectral embedding
    # (faster convergence, more consistent layout than random init)
    reducer_2d = umap.UMAP(
        n_components = 2,
        n_neighbors  = n_neighbors,
        min_dist     = cl_cfg.get("umap_2d_min_dist", 0.1),
        metric       = cl_cfg.get("umap_metric", "cosine"),
        random_state = cl_cfg.get("umap_random_state", 42),
        init         = "spectral",
        low_memory   = True,
        verbose      = False,
    )
    embedding_2d = reducer_2d.fit_transform(matrix)

    return labels_50d, embedding_2d


def _assign_noise_to_nearest(
    labels: List[int],
    matrix: np.ndarray,
) -> List[int]:
    """
    HDBSCAN noise points (label == -1) are assigned to their nearest
    cluster centroid in embedding space.
    If no non-noise clusters exist, all points stay as noise (-1).
    """
    labels_arr = np.array(labels, dtype=np.int32)
    cluster_ids = [l for l in set(labels) if l != -1]

    if not cluster_ids:
        return labels

    # Compute cluster centroids
    centroids: Dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        mask = labels_arr == cid
        centroids[cid] = matrix[mask].mean(axis=0)

    centroid_matrix = np.vstack([centroids[cid] for cid in cluster_ids])
    cluster_id_list = cluster_ids

    noise_mask = labels_arr == -1
    if not noise_mask.any():
        return labels

    noise_embeddings = matrix[noise_mask]

    # Cosine similarity to each centroid
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(noise_embeddings, centroid_matrix)
    nearest = sim.argmax(axis=1)

    labels_arr[noise_mask] = np.array(
        [cluster_id_list[idx] for idx in nearest], dtype=np.int32
    )
    return labels_arr.tolist()


def _annotate_enrichment(
    labels:     List[int],
    index:      List[str],
    cand_by_id: Dict[str, _BaseRecord],
) -> Dict[int, str]:
    """Annotate each cluster with its dominant Pfam domain or function."""
    cluster_domains: Dict[int, List[str]] = defaultdict(list)

    for i, cid in enumerate(index):
        label = int(labels[i])
        c     = cand_by_id.get(cid)
        if c is None:
            continue
        cluster_domains[label].extend(c.pfam_domains)
        # Add track-specific context
        if hasattr(c, "initial_class") and c.initial_class:
            cluster_domains[label].append(c.initial_class)
        if hasattr(c, "spanin_type") and c.spanin_type:
            cluster_domains[label].append(c.spanin_type)

    enrichment: Dict[int, str] = {}
    for label, domains in cluster_domains.items():
        if domains:
            enrichment[label] = Counter(domains).most_common(1)[0][0]
        else:
            enrichment[label] = "unknown"
    return enrichment


# ── File writers ──────────────────────────────────────────────────────────────

def _write_cluster_assignments(
    path:       Path,
    index:      List[str],
    labels:     List[int],
    enrichment: Dict[int, str],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["candidate_id", "cluster_id", "cluster_enrichment"])
        for cid, label in zip(index, labels):
            writer.writerow([cid, label, enrichment.get(label, "")])


def _write_umap_2d(
    path:        Path,
    index:       List[str],
    embedding_2d:np.ndarray,
    labels:      List[int],
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["candidate_id", "umap_x", "umap_y", "cluster_id"])
        for i, cid in enumerate(index):
            writer.writerow([
                cid,
                round(float(embedding_2d[i, 0]), 4),
                round(float(embedding_2d[i, 1]), 4),
                int(labels[i]),
            ])


# ── Snakemake / standalone entry point ───────────────────────────────────────
# snakemake object check must come FIRST — when Snakemake calls this via
# script: directive, __name__ == '__main__', so snakemake takes priority.
if 'snakemake' in dir():
    from pipeline.utils.logging_config import setup_logging
    setup_logging(snakemake.config['paths'].get('log_dir', 'logs'))
    run(snakemake.config)
elif __name__ == '__main__':
    import sys, yaml
    from pipeline.utils.logging_config import setup_logging
    _cfg = yaml.safe_load(open(sys.argv[1]))
    setup_logging(_cfg['paths'].get('log_dir', 'logs'))
    run(_cfg)
