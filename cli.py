"""
cli.py
======
Command-line interface for the prophage lysis module discovery pipeline.

Commands:
  run           — execute the full pipeline (calls Snakemake)
  resume        — resume from a specific module
  check         — validate config and input files before running
  add-target    — add a new target pathogen to the database
  active-learning — train classifiers from wet lab results
  round2-select — re-rank reserve candidates using trained classifiers
  report-status — print current pipeline run status

Install:
  pip install -e .
  prophage_lysis --help
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml

app = typer.Typer(
    name        = "prophage_lysis",
    help        = (
        "Prophage lysis module discovery pipeline — "
        "phenotype-guided, AI-driven enzyme mining from marine Vibrio prophages."
    ),
    add_completion = False,
    no_args_is_help= True,
)


# ── run ───────────────────────────────────────────────────────────────────────

@app.command()
def run(
    config:   Path = typer.Option("config/config.yaml", "--config", "-c",
                                   help="Path to config.yaml"),
    cores:    int  = typer.Option(8,      "--cores",  "-j", help="CPU cores for Snakemake"),
    dryrun:   bool = typer.Option(False,  "--dry-run","-n", help="Dry run — print DAG only"),
    use_conda:bool = typer.Option(False,  "--use-conda",    help="Use per-rule conda envs (not needed if already in conda env)"),
    verbose:  bool = typer.Option(False,  "--verbose", "-v"),
) -> None:
    """Execute the full pipeline from M01 to M09."""
    _validate_config(config)
    cmd = _snakemake_cmd(
        config    = config,
        cores     = cores,
        dryrun    = dryrun,
        use_conda = use_conda,
        verbose   = verbose,
        targets   = [],   # default target = rule all
    )
    typer.echo(f"Running pipeline: {' '.join(cmd)}")
    _run_cmd(cmd)


# ── resume ────────────────────────────────────────────────────────────────────

_MODULE_TARGETS = {
    1: "pharokka_annotation",
    2: "lysis_module_identification",
    3: "gate1_expressibility",
    4: "esm2_embeddings",
    5: "clustering",
    6: "pg_matching",
    7: "redundancy_collapse",
    8: "selection",
    9: "report_generation",
}

@app.command()
def resume(
    from_module: int  = typer.Option(..., "--from-module", "-m",
                                      help="Resume from module number (1-9)"),
    config:  Path = typer.Option("config/config.yaml", "--config", "-c"),
    cores:   int  = typer.Option(8, "--cores", "-j"),
    dryrun:  bool = typer.Option(False, "--dry-run", "-n"),
) -> None:
    """Resume pipeline from a specific module number."""
    if from_module not in _MODULE_TARGETS:
        typer.echo(
            f"[error] Unknown module {from_module}. "
            f"Valid range: 1–9", err=True
        )
        raise typer.Exit(1)

    rule = _MODULE_TARGETS[from_module]
    typer.echo(f"Resuming from module {from_module} (rule: {rule})")

    cmd = _snakemake_cmd(
        config    = config,
        cores     = cores,
        dryrun    = dryrun,
        use_conda = True,
        verbose   = False,
        targets   = [rule],
        force     = True,   # force re-run of this rule even if outputs exist
    )
    _run_cmd(cmd)


# ── check ─────────────────────────────────────────────────────────────────────

@app.command()
def check(
    config: Path = typer.Option("config/config.yaml", "--config", "-c"),
) -> None:
    """Validate config file and check that all required paths exist."""
    typer.echo(f"Checking config: {config}")
    cfg = _validate_config(config)

    checks = {
        "genome_input_dir": cfg["paths"]["genome_input_dir"],
        "pfam_hmm":         cfg["paths"]["pfam_hmm"],
        "pharokka_db":      cfg["paths"]["pharokka_db"],
    }

    all_ok = True
    for name, path in checks.items():
        exists = Path(path).exists()
        status = "✓" if exists else "✗ MISSING"
        typer.echo(f"  {status}  {name}: {path}")
        if not exists:
            all_ok = False

    # Check Pfam is hmmpress'd
    pfam_path = Path(cfg["paths"]["pfam_hmm"])
    h3i = pfam_path.with_suffix(".hmm.h3i")
    if pfam_path.exists() and not h3i.exists():
        typer.echo(
            f"  ✗ Pfam-A.hmm is not hmmpress'd. Run:\n"
            f"    hmmpress {pfam_path}",
            err=True,
        )
        all_ok = False
    elif pfam_path.exists():
        typer.echo(f"  ✓  Pfam-A.hmm hmmpress index: OK")

    # Count genomes
    genome_dir = Path(cfg["paths"]["genome_input_dir"])
    if genome_dir.exists():
        genomes = (
            list(genome_dir.glob("*.fasta")) +
            list(genome_dir.glob("*.fa"))    +
            list(genome_dir.glob("*.fna"))
        )
        typer.echo(f"  ✓  Input genomes found: {len(genomes)}")
        if not genomes:
            typer.echo(
                "  ✗ No FASTA files in genome input directory!", err=True
            )
            all_ok = False

    if all_ok:
        typer.echo("\n[OK] All checks passed — ready to run.")
    else:
        typer.echo("\n[FAIL] Fix issues above before running.", err=True)
        raise typer.Exit(1)


# ── add-target ────────────────────────────────────────────────────────────────

@app.command(name="add-target")
def add_target(
    pathogen_id:    str  = typer.Option(..., "--id",          help="Unique pathogen ID (snake_case)"),
    display_name:   str  = typer.Option(..., "--name",        help="Display name, e.g. 'Vibrio vulnificus'"),
    species:        str  = typer.Option(..., "--species",     help="Scientific species name"),
    gram_stain:     str  = typer.Option(..., "--gram-stain",  help="'positive' or 'negative'"),
    pg_chemotype:   str  = typer.Option(..., "--pg-chemotype",
                                         help=(
                                             "PG chemistry type. One of:\n"
                                             "  gram_negative_dap\n"
                                             "  gram_negative_dap_om_barrier\n"
                                             "  gram_positive_lys\n"
                                             "  gram_positive_dap"
                                         )),
    aquaculture_host: str = typer.Option("", "--aquaculture-host", help="Target aquaculture species"),
    notes:          str  = typer.Option("", "--notes",        help="Additional notes"),
    config:         Path = typer.Option("config/config.yaml", "--config"),
) -> None:
    """Add a new target pathogen to the database."""
    from pipeline.utils.pg_database import PathogenDatabase

    cfg     = _validate_config(config)
    db_path = cfg.get("targets", {}).get("pathogen_db", "targets/pathogen_db.yaml")
    db      = PathogenDatabase(db_path)

    try:
        db.add_target(
            pathogen_id     = pathogen_id,
            display_name    = display_name,
            species         = species,
            gram_stain      = gram_stain,
            pg_chemotype    = pg_chemotype,
            aquaculture_host= aquaculture_host,
            notes           = notes,
        )
        typer.echo(
            f"[OK] Added pathogen '{pathogen_id}' ({display_name}) "
            f"to {db_path}"
        )
        typer.echo(f"  PG chemotype: {pg_chemotype}")
        typer.echo(f"  Gram stain:   {gram_stain}")
        typer.echo(
            "\nRe-run M06 to score existing endolysins against the new target:\n"
            f"  prophage_lysis resume --from-module 6 --config {config}"
        )
    except ValueError as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)


# ── active-learning ───────────────────────────────────────────────────────────

@app.command(name="active-learning")
def active_learning(
    results: Path = typer.Option(...,   "--results", "-r",
                                  help="Wet lab results CSV file"),
    config:  Path = typer.Option("config/config.yaml", "--config", "-c"),
    output:  Path = typer.Option("data/active_learning", "--output", "-o"),
    append:  bool = typer.Option(False, "--append",
                                  help="Append to previous wet lab results (multi-round)"),
) -> None:
    """Train pathogen activity classifiers from wet lab results (M10)."""
    from pipeline.active_learning.train import train

    if not results.exists():
        typer.echo(f"[error] Results file not found: {results}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Training classifiers from: {results}")
    if append:
        typer.echo("  Mode: APPEND (combining with previous rounds)")

    _validate_config(config)

    meta = train(
        results_csv = str(results),
        config_path = str(config),
        output_dir  = str(output),
        append      = append,
    )

    typer.echo(f"\n[OK] Trained {len(meta)} pathogen classifiers:")
    for pid, m in meta.items():
        auc = f"  CV AUC={m['cv_auc']:.3f}" if m.get("cv_auc") else ""
        typer.echo(f"  {pid}: {m['model_type']} n={m['n_train']}{auc}")

    typer.echo(
        f"\nNext step — re-rank reserve candidates:\n"
        f"  prophage_lysis round2-select --al-dir {output} --config {config}"
    )


# ── round2-select ─────────────────────────────────────────────────────────────

@app.command(name="round2-select")
def round2_select(
    al_dir:  Path = typer.Option("data/active_learning", "--al-dir"),
    config:  Path = typer.Option("config/config.yaml",   "--config", "-c"),
    output:  Path = typer.Option("data/active_learning", "--output", "-o"),
) -> None:
    """Re-rank reserve candidates with trained classifiers (M11)."""
    from pipeline.active_learning.predict import predict_and_rerank

    _validate_config(config)

    if not (al_dir / "model_metadata.json").exists():
        typer.echo(
            f"[error] No trained models found in {al_dir}. "
            f"Run `prophage_lysis active-learning` first.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Re-ranking reserve candidates using models from: {al_dir}")
    predict_and_rerank(
        config_path = str(config),
        al_dir      = str(al_dir),
        output_dir  = str(output),
    )
    typer.echo(
        f"\n[OK] Round 2 candidates written to: "
        f"{output}/round2_candidates.csv"
    )


# ── report-status ─────────────────────────────────────────────────────────────

@app.command(name="report-status")
def report_status(
    config:  Path = typer.Option("config/config.yaml", "--config", "-c"),
) -> None:
    """Print current pipeline run status — which modules are complete."""
    cfg     = _validate_config(config)
    int_dir = Path(cfg["paths"]["intermediate_dir"])
    out_dir = Path(cfg["paths"]["output_dir"])

    checkpoints = [
        (1, "M01 Pharokka",      int_dir / "01_pharokka"    / "all_proteins.faa"),
        (2, "M02 Lysis modules", int_dir / "02_lysis_modules" / "candidates.json"),
        (3, "M03 Gate 1",        int_dir / "03_gate1"        / "candidates_passing.json"),
        (4, "M04 Embeddings",    int_dir / "04_embeddings"   / "embedding_matrix.npy"),
        (5, "M05 Clustering",    int_dir / "05_clusters"     / "cluster_summary.json"),
        (6, "M06 PG matching",   int_dir / "06_pg_matching"  / "pg_scores.tsv"),
        (7, "M07 Redundancy",    int_dir / "07_redundancy"   / "gate3_results.tsv"),
        (8, "M08 Selection",     int_dir / "08_selection"    / "selection_summary.json"),
        (9, "M09 Report",        out_dir / "priority_list.csv"),
        (10,"M10 AL training",   Path("data/active_learning") / "model_metadata.json"),
        (11,"M11 Round 2",       Path("data/active_learning") / "round2_candidates.csv"),
    ]

    typer.echo("\nPipeline status:")
    for n, name, checkpoint in checkpoints:
        done = checkpoint.exists()
        mark = "✓" if done else "○"
        size = ""
        if done and checkpoint.is_file():
            sz = checkpoint.stat().st_size
            size = f"  ({sz / 1024:.0f} KB)" if sz > 1024 else f"  ({sz} B)"
        typer.echo(f"  {mark}  M{n:02d}  {name}{size}")

    # Summary from pipeline_summary.json if available
    summary_path = out_dir / "pipeline_summary.json"
    if summary_path.exists():
        import json
        s = json.loads(summary_path.read_text())
        typer.echo(f"\nResults summary:")
        typer.echo(f"  Priority candidates: {s.get('priority_count', '?')}")
        typer.echo(f"  Reserve candidates:  {s.get('reserve_count', '?')}")
        typer.echo(f"  Novel endolysins:    {s.get('novel_endolysins', '?')}")
        typer.echo(f"  Complete modules:    {s.get('complete_modules', '?')}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_config(config_path: Path) -> dict:
    if not Path(config_path).exists():
        typer.echo(f"[error] Config not found: {config_path}", err=True)
        raise typer.Exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def _snakemake_cmd(
    config:    Path,
    cores:     int,
    dryrun:    bool,
    use_conda: bool,
    verbose:   bool,
    targets:   list,
    force:     bool = False,
) -> list:
    cmd = [
        "snakemake",
        "--configfile", str(config),
        "--cores",      str(cores),
        "--rerun-incomplete",
        "--keep-going",
        "--printshellcmds",
    ]
    if dryrun:    cmd.append("--dry-run")
    if use_conda: cmd.append("--use-conda")
    if verbose:   cmd.append("--verbose")
    if force:     cmd.extend(["--forcerun"] + targets)
    cmd.extend(targets)
    return cmd


def _run_cmd(cmd: list) -> None:
    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            typer.echo(
                f"\n[error] Pipeline exited with code {result.returncode}. "
                f"Check logs in data/output/logs/",
                err=True,
            )
            raise typer.Exit(result.returncode)
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user.")
        raise typer.Exit(130)


if __name__ == "__main__":
    app()
