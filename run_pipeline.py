"""
run_pipeline.py
===============
Standalone sequential pipeline runner.
Alternative to Snakemake for single-machine use without workflow engine overhead.
Useful for debugging individual modules and for CI/CD testing.

Usage:
  python run_pipeline.py --config config/config.yaml
  python run_pipeline.py --config config/config.yaml --from 3 --to 5
  python run_pipeline.py --config config/config.yaml --only 7
  python run_pipeline.py --config config/config.yaml --dry-run
"""

from __future__ import annotations
import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import yaml

# Ensure pipeline package is importable
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils.logging_config import setup_logging
import logging

log = logging.getLogger("runner")


# ── Module registry ───────────────────────────────────────────────────────────

def _get_module(n: int):
    """Lazy import each module to avoid loading all heavy deps at startup."""
    if n == 1:
        from pipeline.modules import m01_pharokka as m; return m
    if n == 2:
        from pipeline.modules import m02_lysis_modules as m; return m
    if n == 3:
        from pipeline.modules import m03_gate1 as m; return m
    if n == 4:
        from pipeline.modules import m04_embeddings as m; return m
    if n == 5:
        from pipeline.modules import m05_clustering as m; return m
    if n == 6:
        from pipeline.modules import m06_pg_matching as m; return m
    if n == 7:
        from pipeline.modules import m07_redundancy as m; return m
    if n == 8:
        from pipeline.modules import m08_selection as m; return m
    if n == 9:
        from pipeline.modules import m09_report as m; return m
    raise ValueError(f"Unknown module number: {n}")


MODULE_NAMES = {
    1: "M01 Pharokka annotation",
    2: "M02 Lysis module identification",
    3: "M03 Gate 1 expressibility filter",
    4: "M04 ESM-2 embeddings",
    5: "M05 UMAP + HDBSCAN clustering",
    6: "M06 PG chemistry matching",
    7: "M07 Redundancy collapse",
    8: "M08 MaxMin selection",
    9: "M09 Report generation",
}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_exists(n: int, cfg: dict) -> bool:
    """True if this module's primary output already exists."""
    inter = cfg["paths"]["intermediate_dir"]
    out   = cfg["paths"]["output_dir"]
    checkpoints = {
        1: f"{inter}/01_pharokka/all_proteins.faa",
        2: f"{inter}/02_lysis_modules/candidates.json",
        3: f"{inter}/03_gate1/candidates_passing.json",
        4: f"{inter}/04_embeddings/embedding_matrix.npy",
        5: f"{inter}/05_clusters/cluster_summary.json",
        6: f"{inter}/06_pg_matching/pg_scores.tsv",
        7: f"{inter}/07_redundancy/gate3_results.tsv",
        8: f"{inter}/08_selection/selection_summary.json",
        9: f"{out}/priority_list.csv",
    }
    p = checkpoints.get(n)
    return Path(p).exists() if p else False


# ── Runner ────────────────────────────────────────────────────────────────────

def run_pipeline(
    cfg:        dict,
    modules:    List[int],
    dry_run:    bool = False,
    resume:     bool = True,
) -> None:
    """
    Run specified modules in order.

    Args:
        cfg:     loaded config dict
        modules: list of module numbers to execute [1..9]
        dry_run: print plan without executing
        resume:  skip modules whose checkpoint output already exists
    """
    log.info(f"Pipeline run: modules {modules[0]}–{modules[-1]}")
    log.info(f"  dry_run={dry_run}  resume={resume}")

    results: dict = {}
    total_start = time.time()

    for n in modules:
        name = MODULE_NAMES.get(n, f"M{n:02d}")

        # Skip M06 if PG matching disabled
        if n == 6 and not cfg.get("pg_matching", {}).get("enabled", False):
            log.info(f"  [SKIP] {name} — pg_matching.enabled: false")
            continue

        if resume and _checkpoint_exists(n, cfg):
            log.info(f"  [SKIP] {name} — checkpoint exists")
            continue

        if dry_run:
            log.info(f"  [DRY]  {name}")
            continue

        log.info(f"  [RUN]  {name}")
        t0 = time.time()

        try:
            module = _get_module(n)
            module.run(cfg)
            elapsed = time.time() - t0
            log.info(f"  [OK]   {name} — {elapsed:.1f}s")
            results[n] = {"status": "ok", "elapsed": elapsed}

        except Exception as e:
            elapsed = time.time() - t0
            log.error(f"  [FAIL] {name} — {e}")
            log.debug(traceback.format_exc())
            results[n] = {"status": "fail", "error": str(e), "elapsed": elapsed}
            log.error(
                f"Pipeline stopped at module {n}. "
                f"Fix the issue and resume with: "
                f"python run_pipeline.py --config ... --from {n}"
            )
            sys.exit(1)

    total = time.time() - total_start
    log.info(f"Pipeline complete — total time: {total:.1f}s")

    # Summary
    n_ok   = sum(1 for r in results.values() if r["status"] == "ok")
    n_skip = sum(1 for n in modules if n not in results)
    log.info(f"  Executed: {n_ok}  Skipped (checkpoint): {n_skip}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="prophage_lysis — standalone sequential pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config config/config.yaml
  python run_pipeline.py --config config/config.yaml --from 3
  python run_pipeline.py --config config/config.yaml --from 4 --to 7
  python run_pipeline.py --config config/config.yaml --only 5
  python run_pipeline.py --config config/config.yaml --dry-run
  python run_pipeline.py --config config/config.yaml --no-resume --from 3
        """,
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--from", dest="from_module", type=int, default=1, metavar="N",
        help="Start from module N (default: 1)",
    )
    parser.add_argument(
        "--to", dest="to_module", type=int, default=9, metavar="N",
        help="Stop after module N (default: 9)",
    )
    parser.add_argument(
        "--only", type=int, default=None, metavar="N",
        help="Run only module N (overrides --from/--to)",
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print plan without executing",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-run all modules even if checkpoint outputs exist",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Setup logging
    setup_logging(
        log_dir = cfg["paths"].get("log_dir", "logs"),
        level   = args.log_level,
    )

    # Determine modules to run
    if args.only is not None:
        modules = [args.only]
    else:
        modules = list(range(args.from_module, args.to_module + 1))

    for n in modules:
        if n not in MODULE_NAMES:
            print(f"[error] Invalid module number: {n}. Valid range: 1–9")
            sys.exit(1)

    run_pipeline(
        cfg     = cfg,
        modules = modules,
        dry_run = args.dry_run,
        resume  = not args.no_resume,
    )


if __name__ == "__main__":
    main()
