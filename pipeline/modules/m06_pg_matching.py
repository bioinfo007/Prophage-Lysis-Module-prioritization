"""
m06_pg_matching.py
==================
Module 06: Peptidoglycan chemistry matching for target pathogen specificity.

Optional — controlled by config: pg_matching.enabled: true/false
New targets can be added via: prophage_lysis add-target ...

Only runs on endolysin track — holins and spanins are not scored
(their activity against pathogens is not determined by PG chemistry).

Input:  data/intermediate/03_gate1/candidates_passing.json
        targets/pathogen_db.yaml
Output: data/intermediate/06_pg_matching/pg_scores.tsv
        data/intermediate/03_gate1/candidates_passing.json (updated)
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List

from pipeline.utils.data_model import (
    EndolysínRecord, _BaseRecord,
    load_candidates, save_candidates, split_by_track,
)
from pipeline.utils.pg_database import PathogenDatabase

log = logging.getLogger("m06_pg_matching")


def run(cfg: dict) -> None:
    paths    = cfg["paths"]
    pg_cfg   = cfg.get("pg_matching", {})

    if not pg_cfg.get("enabled", False):
        log.info("PG matching disabled (pg_matching.enabled: false) — skipping M06")
        return

    db_path  = cfg.get("targets", {}).get(
        "pathogen_db", "targets/pathogen_db.yaml"
    )
    in_dir   = Path(paths["intermediate_dir"]) / "03_gate1"
    out_dir  = Path(paths["intermediate_dir"]) / "06_pg_matching"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path  = in_dir / "candidates_passing.json"
    candidates = load_candidates(str(cand_path))
    tracks     = split_by_track(candidates)

    db = PathogenDatabase(db_path)
    if not db.pathogen_ids():
        log.warning("Pathogen DB is empty — no PG scoring performed")
        return

    log.info(
        f"PG matching: {len(db.pathogen_ids())} target pathogens | "
        f"{len(tracks['endolysin'])} endolysins"
    )

    rows = []
    for c in tracks["endolysin"]:
        if not isinstance(c, EndolysínRecord):
            continue

        domain_type = c.catalytic_domain_type
        scores      = db.score_all_pathogens(domain_type)
        c.set_pg_compatibility(scores)

        row = {"candidate_id": c.candidate_id, "domain_type": domain_type}
        row.update({f"score_{pid}": s for pid, s in scores.items()})
        rows.append(row)

    # Write TSV
    if rows:
        with open(out_dir / "pg_scores.tsv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    save_candidates(candidates, str(cand_path))

    # Log coverage summary
    all_pg_scores = [
        c.get_pg_compatibility()
        for c in tracks["endolysin"]
        if isinstance(c, EndolysínRecord) and c.pg_compatibility
    ]
    if all_pg_scores:
        cov = db.coverage_summary(all_pg_scores)
        log.info("PG compatibility coverage (score ≥ 2):")
        for pid, count in cov.items():
            log.info(f"  {pid}: {count} endolysins")

    log.info(f"M06 complete — {len(rows)} endolysins scored")


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
