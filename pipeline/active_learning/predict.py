"""
active_learning/predict.py
==========================
Module 11: Re-rank reserve candidates using trained activity classifiers.

Loads models trained in train.py and scores all reserve endolysins.
The re-ranked list feeds into round 2 selection, prioritizing candidates
that fill pathogen coverage gaps left by round 1.

Usage via CLI:
  prophage_lysis round2-select --al-dir data/active_learning --config config.yaml
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline.utils.data_model import (
    EndolysínRecord, _BaseRecord,
    load_candidates, save_candidates, split_by_track,
)

log = logging.getLogger("active_learning.predict")


def predict_and_rerank(
    config_path: str,
    al_dir:      str,
    output_dir:  str,
) -> None:
    """
    Score all reserve endolysins with trained pathogen classifiers.
    Write re-ranked round 2 selection list.
    """
    import yaml
    import pandas as pd

    cfg      = yaml.safe_load(open(config_path))
    al_path  = Path(al_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models_dir   = al_path / "models"
    meta_path    = al_path / "model_metadata.json"
    emb_dir      = Path(cfg["paths"]["intermediate_dir"]) / "04_embeddings"
    cand_path    = Path(cfg["paths"]["intermediate_dir"]) / "03_gate1" / "candidates_passing.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No model metadata at {meta_path}. "
            f"Run `prophage_lysis active-learning` first."
        )

    model_meta  = json.loads(meta_path.read_text())
    candidates  = load_candidates(str(cand_path))
    tracks      = split_by_track(candidates)

    # Reserve endolysins only
    reserve_endos = [
        c for c in tracks["endolysin"]
        if c.final_status == "reserve"
    ]

    if not reserve_endos:
        log.info("No reserve endolysins — nothing to re-rank")
        return

    log.info(
        f"Re-ranking {len(reserve_endos)} reserve endolysins | "
        f"{len(model_meta)} pathogen models"
    )

    # Load embedding matrix
    full_matrix = np.load(emb_dir / "embedding_matrix.npy")
    full_index  = json.loads((emb_dir / "embedding_index.json").read_text())
    id_to_row   = {cid: i for i, cid in enumerate(full_index)}

    # Load models
    models: Dict[str, object] = {}
    for pathogen, meta in model_meta.items():
        model_path = Path(meta["model_path"])
        if model_path.exists():
            with open(model_path, "rb") as f:
                models[pathogen] = pickle.load(f)
            log.info(
                f"  Loaded {pathogen}: {meta['model_type']} "
                f"(CV AUC={meta.get('cv_auc', 'N/A')})"
            )
        else:
            log.warning(f"  Model file missing: {model_path}")

    if not models:
        raise RuntimeError("No trained models could be loaded.")

    # Score each reserve endolysin
    scored_rows = []

    for c in reserve_endos:
        if c.candidate_id not in id_to_row:
            log.debug(f"  No embedding for {c.candidate_id} — skipping")
            continue

        emb = full_matrix[id_to_row[c.candidate_id]].reshape(1, -1).astype(np.float32)
        scores: Dict[str, float] = {}

        for pathogen, model in models.items():
            try:
                prob = float(model.predict_proba(emb)[0, 1])
            except Exception:
                prob = float(model.predict(emb)[0])
            scores[pathogen] = round(prob, 4)

        # Store scores on candidate
        c.set_pathogen_scores(scores)

        # Composite re-rank score:
        # Mean predicted probability across all pathogens (equal weight)
        composite = np.mean(list(scores.values()))

        scored_rows.append({
            "candidate_id":  c.candidate_id,
            "genome_id":     c.genome_id,
            "module_id":     c.module_id or "",
            "module_complete": c.module_complete,
            "cluster_id":    c.cluster_id,
            "composite_al_score": round(float(composite), 4),
            **{f"prob_{pid}": s for pid, s in scores.items()},
            "novelty_flag":  c.novelty_flag or "",
            "sequence":      c.sequence,
        })

    if not scored_rows:
        log.warning("No reserve endolysins could be scored")
        return

    # Sort by composite score descending
    scored_rows.sort(key=lambda r: r["composite_al_score"], reverse=True)

    # Write re-ranked list
    reranked_path = out_path / "round2_candidates.csv"
    import csv
    with open(reranked_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scored_rows[0].keys())
        writer.writeheader()
        writer.writerows(scored_rows)

    log.info(f"Re-ranked list written: {reranked_path}")

    # Update candidates with AL scores
    save_candidates(candidates, str(cand_path))

    # Summary: which pathogens are well covered vs. gaps
    log.info("Predicted coverage in reserve pool (top-20 candidates):")
    top20_scores = [
        {pid: r[f"prob_{pid}"] for pid in models
         if f"prob_{pid}" in r}
        for r in scored_rows[:20]
    ]
    for pathogen in models:
        key       = f"prob_{pathogen}"
        top_probs = [r.get(key, 0) for r in scored_rows[:20]]
        high_conf = sum(1 for p in top_probs if p > 0.6)
        log.info(
            f"  {pathogen}: {high_conf}/20 top candidates "
            f"with P(active) > 0.60"
        )

    log.info(
        f"Round 2 selection ready — {len(scored_rows)} candidates scored | "
        f"top candidate: {scored_rows[0]['candidate_id']} "
        f"(score={scored_rows[0]['composite_al_score']})"
    )


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <config.yaml>")
        sys.exit(1)
    from pipeline.utils.logging_config import setup_logging
    import yaml
    cfg = yaml.safe_load(open(sys.argv[1]))
    setup_logging(cfg["paths"].get("log_dir", "logs"))
    predict_and_rerank(
        config_path = sys.argv[1],
        al_dir      = "data/active_learning",
        output_dir  = "data/active_learning",
    )
