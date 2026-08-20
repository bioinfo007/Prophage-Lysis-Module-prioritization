"""
active_learning/train.py
========================
Module 10: Train pathogen-specific activity classifiers on wet lab results.

Takes wet lab activity data (CSV from expression round) and ESM-2 embeddings.
Trains one lightweight model per target pathogen using endolysin embedding
vectors as features and binary/continuous activity measurements as labels.

At 20–50 labeled examples, logistic regression and random forest on 1280-dim
ESM-2 features generalize reliably. No deep learning required at this scale.

Usage via CLI:
  prophage_lysis active-learning --results wetlab_round1.csv --config config.yaml
  prophage_lysis active-learning --results wetlab_round2.csv --config config.yaml --append
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("active_learning.train")


# ── Wet lab results schema ────────────────────────────────────────────────────
# Expected CSV columns:
#   candidate_id, pathogen_id, mic_ug_ml, kill_pct_6h, active (0/1)
#
# "active" is the primary label:
#   1 = confirmed activity (MIC < threshold or kill% > threshold)
#   0 = no activity
# Continuous values (mic, kill_pct) are used for regression if > 10 actives.

_MIC_ACTIVE_THRESHOLD   = 50.0    # ug/mL — below this = active
_KILL_ACTIVE_THRESHOLD  = 50.0    # % kill at 6h — above this = active
_MIN_SAMPLES_REGRESSION = 15      # use regression only if enough actives


def train(
    results_csv: str,
    config_path: str,
    output_dir:  str,
    append:      bool = False,
) -> None:
    """
    Train/retrain activity classifiers from wet lab results CSV.
    Saves one model file per pathogen to output_dir/models/.
    """
    import yaml
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    cfg = yaml.safe_load(open(config_path))
    emb_dir  = Path(cfg["paths"]["intermediate_dir"]) / "04_embeddings"
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Load embedding index and matrix
    full_matrix = np.load(emb_dir / "embedding_matrix.npy")
    full_index  = json.loads((emb_dir / "embedding_index.json").read_text())
    id_to_row   = {cid: i for i, cid in enumerate(full_index)}

    # Load wet lab results
    df = pd.read_csv(results_csv)
    log.info(f"Loaded {len(df)} wet lab records from {results_csv}")

    # If appending, load any previous round data
    history_path = out_dir / "wetlab_history.csv"
    if append and history_path.exists():
        history = pd.read_csv(history_path)
        df = pd.concat([history, df], ignore_index=True).drop_duplicates(
            subset=["candidate_id", "pathogen_id"]
        )
        log.info(f"Combined with history: {len(df)} total records")

    df.to_csv(history_path, index=False)

    # ── Train one model per pathogen ──────────────────────────────────────────
    pathogens    = df["pathogen_id"].unique()
    model_meta   = {}

    for pathogen in pathogens:
        pdf  = df[df["pathogen_id"] == pathogen].copy()

        # Build label vector
        if "active" in pdf.columns:
            pdf["label"] = pdf["active"].astype(int)
        else:
            # Infer from MIC/kill_pct
            cond1 = (pdf.get("mic_ug_ml", np.inf) < _MIC_ACTIVE_THRESHOLD)
            cond2 = (pdf.get("kill_pct_6h", 0) > _KILL_ACTIVE_THRESHOLD)
            pdf["label"] = ((cond1 | cond2)).astype(int)

        # Collect embeddings
        X_rows, y_rows = [], []
        for _, row in pdf.iterrows():
            cid = row["candidate_id"]
            if cid not in id_to_row:
                log.debug(f"  No embedding for {cid} — skipping")
                continue
            X_rows.append(full_matrix[id_to_row[cid]])
            y_rows.append(int(row["label"]))

        if len(X_rows) < 5:
            log.warning(
                f"  {pathogen}: only {len(X_rows)} labeled samples — "
                f"skipping (need ≥ 5)"
            )
            continue

        X = np.vstack(X_rows).astype(np.float32)
        y = np.array(y_rows, dtype=np.int32)

        n_active   = int(y.sum())
        n_inactive = int((1 - y).sum())
        log.info(
            f"  {pathogen}: n={len(y)} | active={n_active} | "
            f"inactive={n_inactive}"
        )

        # Choose model: RandomForest for ≥ 20 samples, LogisticRegression otherwise
        if len(y) >= 20:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                )),
            ])
            model_type = "RandomForest"
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                )),
            ])
            model_type = "LogisticRegression"

        clf.fit(X, y)

        # Cross-validation if enough samples
        if len(y) >= 10:
            cv_scores = cross_val_score(
                clf, X, y, cv=min(5, n_active, n_inactive, len(y) // 2),
                scoring="roc_auc", n_jobs=-1,
            )
            cv_auc = float(np.mean(cv_scores))
            log.info(f"    CV AUC: {cv_auc:.3f} ± {cv_scores.std():.3f}")
        else:
            cv_auc = None
            log.info("    CV skipped (n < 10)")

        # Save model
        model_path = models_dir / f"{pathogen}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        model_meta[pathogen] = {
            "model_path":  str(model_path),
            "model_type":  model_type,
            "n_train":     len(y),
            "n_active":    n_active,
            "n_inactive":  n_inactive,
            "cv_auc":      cv_auc,
            "round":       results_csv,
        }
        log.info(f"    Model saved: {model_path.name}")

    # Save metadata
    meta_path = out_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(model_meta, indent=2))

    log.info(
        f"Active learning training complete — {len(model_meta)} pathogen models"
    )
    return model_meta


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python train.py <results.csv> <config.yaml> [--append]")
        sys.exit(1)
    from pipeline.utils.logging_config import setup_logging
    import yaml
    cfg = yaml.safe_load(open(sys.argv[2]))
    setup_logging(cfg["paths"].get("log_dir", "logs"))
    append = "--append" in sys.argv
    train(
        results_csv = sys.argv[1],
        config_path = sys.argv[2],
        output_dir  = "data/active_learning",
        append      = append,
    )
