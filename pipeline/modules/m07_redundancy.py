"""
m07_redundancy.py
=================
Module 07: Per-track redundancy collapse.

Groups near-identical candidates using cosine similarity on ESM-2 embeddings.
Similarity computed per track — endolysins are never collapsed against holins.

For large candidate pools (N > 5000), block cosine similarity avoids OOM.
Representative selection uses composite criterion:
  module_complete bonus + pLDDT (if available) + fewest Gate 1 flags + lowest MW

Input:  data/intermediate/03_gate1/candidates_passing.json
        data/intermediate/04_embeddings/embedding_matrix.npy
        data/intermediate/04_embeddings/embedding_index.json
        data/intermediate/02_lysis_modules/modules.json
Output: data/intermediate/07_redundancy/{track}_redundancy_clusters.json
        data/intermediate/07_redundancy/{track}_similarity_edges.tsv
        data/intermediate/07_redundancy/gate3_results.tsv
        data/intermediate/03_gate1/candidates_passing.json (updated)
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import numpy as np
import networkx as nx

from pipeline.utils.data_model import (
    _BaseRecord, EndolysínRecord,
    load_candidates, save_candidates, split_by_track,
)
from pipeline.utils.numba_kernels import block_cosine_similarity

log = logging.getLogger("m07_redundancy")

_BLOCK_THRESHOLD = 5000   # use block computation above this size


def run(cfg: dict) -> None:
    paths  = cfg["paths"]
    g3_cfg = cfg["gate3"]

    in_dir  = Path(paths["intermediate_dir"]) / "03_gate1"
    emb_dir = Path(paths["intermediate_dir"]) / "04_embeddings"
    out_dir = Path(paths["intermediate_dir"]) / "07_redundancy"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path  = in_dir / "candidates_passing.json"
    candidates = load_candidates(str(cand_path))

    full_matrix = np.load(emb_dir / "embedding_matrix.npy")
    full_index  = json.loads((emb_dir / "embedding_index.json").read_text())
    id_to_row   = {cid: i for i, cid in enumerate(full_index)}

    similarity_threshold = g3_cfg.get("similarity_threshold", 0.92)
    criterion            = g3_cfg.get("representative_criterion", "composite")

    tracks   = split_by_track(candidates)
    all_results: List[dict] = []

    n_rep = n_collapsed = 0

    for track_name, track_cands in tracks.items():
        if not track_cands:
            continue

        log.info(
            f"  Redundancy collapse [{track_name}]: "
            f"{len(track_cands)} candidates | threshold={similarity_threshold}"
        )

        # Get embedding sub-matrix for this track
        track_ids   = [c.candidate_id for c in track_cands if c.candidate_id in id_to_row]
        track_rows  = np.vstack([full_matrix[id_to_row[cid]] for cid in track_ids]).astype(np.float32)
        cand_by_id  = {c.candidate_id: c for c in track_cands}

        N = len(track_ids)

        if N == 0:
            continue

        # Record max similarity to any other candidate (before collapsing)
        if N < _BLOCK_THRESHOLD:
            from pipeline.utils.numba_kernels import cosine_similarity_matrix
            sim_matrix = cosine_similarity_matrix(track_rows)
            np.fill_diagonal(sim_matrix, 0.0)
            max_sims = sim_matrix.max(axis=1)
            for i, cid in enumerate(track_ids):
                c = cand_by_id.get(cid)
                if c:
                    c.max_similarity = round(float(max_sims[i]), 4)
            np.fill_diagonal(sim_matrix, 1.0)

            # Build graph from similarity matrix
            G = nx.Graph()
            G.add_nodes_from(range(N))
            ii, jj = np.where(sim_matrix > similarity_threshold)
            for i, j in zip(ii, jj):
                if i < j:
                    G.add_edge(int(i), int(j))

        else:
            # Block computation for large N
            log.info(f"    N={N} > {_BLOCK_THRESHOLD} — using block similarity")
            edges = block_cosine_similarity(track_rows, similarity_threshold)
            G = nx.Graph()
            G.add_nodes_from(range(N))
            for row in edges:
                G.add_edge(int(row[0]), int(row[1]))
            # Max similarity approximation (from block edges)
            max_sims = np.zeros(N, dtype=np.float32)
            for row in edges:
                i, j, s = int(row[0]), int(row[1]), float(row[2])
                max_sims[i] = max(max_sims[i], s)
                max_sims[j] = max(max_sims[j], s)
            for i, cid in enumerate(track_ids):
                c = cand_by_id.get(cid)
                if c:
                    c.max_similarity = round(float(max_sims[i]), 4)

        log.info(
            f"    Graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

        components = list(nx.connected_components(G))
        n_singletons = sum(1 for comp in components if len(comp) == 1)
        n_groups     = len(components) - n_singletons

        log.info(
            f"    {len(components)} components | "
            f"{n_singletons} singletons | {n_groups} redundancy groups"
        )

        cluster_records = []

        for comp_id, component in enumerate(components):
            member_ids = [track_ids[i] for i in component]
            members    = [cand_by_id[cid] for cid in member_ids if cid in cand_by_id]

            if not members:
                continue

            rep = _pick_representative(members, criterion)

            for m in members:
                m.redundancy_cluster = comp_id
                if m.candidate_id == rep.candidate_id:
                    m.is_representative = True
                    m.gate3_status      = "representative"
                    n_rep += 1
                else:
                    m.is_representative = False
                    m.gate3_status      = "collapsed"
                    m.similar_to        = rep.candidate_id
                    n_collapsed        += 1

            cluster_records.append({
                "track":          track_name,
                "cluster_id":     comp_id,
                "size":           len(members),
                "representative": rep.candidate_id,
                "members":        [m.candidate_id for m in members],
                "criterion":      criterion,
            })

        # Write per-track redundancy clusters
        (out_dir / f"{track_name}_redundancy_clusters.json").write_text(
            json.dumps(cluster_records, indent=2)
        )
        all_results.extend(cluster_records)

    # Update candidates
    save_candidates(candidates, str(cand_path))

    # Write combined Gate 3 results TSV
    _write_gate3_results(candidates, out_dir / "gate3_results.tsv")

    log.info(
        f"M07 complete — {n_rep} representatives | {n_collapsed} collapsed"
    )


# ── Representative selection ──────────────────────────────────────────────────

def _pick_representative(
    members:   List[_BaseRecord],
    criterion: str,
) -> _BaseRecord:
    if len(members) == 1:
        return members[0]

    def composite_score(c: _BaseRecord) -> float:
        # Module completeness bonus — strongly prefer representatives from complete modules
        module_bonus  = 0.3 if c.module_complete else 0.0
        # pLDDT — default 70 if not available
        plddt_score   = (getattr(c, "mean_plddt", None) or 70.0) / 100.0
        # Penalty per Gate 1 flag
        flag_penalty  = len(c.gate1_flags) * 0.08
        # MW penalty — prefer smaller
        mw_penalty    = max(0, (c.mw_kda or 50.0) - 20.0) / 100.0
        # CAI bonus
        cai_bonus     = (c.cai_score or 0.5) * 0.1
        return module_bonus + plddt_score + cai_bonus - flag_penalty - mw_penalty

    if criterion == "best_expressibility":
        return min(members, key=lambda c: (len(c.gate1_flags), c.mw_kda or 50))
    else:   # composite (default)
        return max(members, key=composite_score)


def _write_gate3_results(candidates: List[_BaseRecord], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "candidate_id", "track", "gate3_status",
            "redundancy_cluster", "is_representative",
            "similar_to", "max_similarity",
        ])
        for c in candidates:
            writer.writerow([
                c.candidate_id, c.track,
                c.gate3_status or "",
                c.redundancy_cluster or "",
                c.is_representative or "",
                c.similar_to or "",
                c.max_similarity or "",
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
