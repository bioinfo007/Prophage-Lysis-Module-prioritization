"""
Snakefile — prophage lysis module discovery pipeline v2.0.0

Usage:
  snakemake --configfile config/config.yaml --cores 8 -n          # dry-run
  snakemake --configfile config/config.yaml --cores 8 --use-conda # full run

  # HPC (SLURM)
  snakemake --configfile config/config.yaml --cores 200 --use-conda \
    --cluster "sbatch --mem={resources.mem_mb}M \
               --cpus-per-task={threads} \
               --time={resources.runtime} \
               --output=logs/slurm_%j.out"

  # Active learning (triggered manually after wet lab)
  snakemake active_learning_train \
    --configfile config/config.yaml \
    --config wetlab_results=data/wetlab/round1.csv --cores 4

  # Utility
  snakemake --configfile config/config.yaml clean_intermediate
  snakemake --configfile config/config.yaml dag_png
"""

import sys
from pathlib import Path

# Make the pipeline package importable when Snakemake executes script: rules
sys.path.insert(0, str(Path(workflow.basedir)))

configfile: "config/config.yaml"

INTERMEDIATE = config["paths"]["intermediate_dir"]
OUTPUT       = config["paths"]["output_dir"]
INPUT_GENOME = config["paths"]["genome_input_dir"]
LOG_DIR      = config["paths"].get("log_dir", f"{OUTPUT}/logs")

PG_ENABLED = config.get("pg_matching", {}).get("enabled", False)


# ── Default target ─────────────────────────────────────────────────────────────────────

rule all:
    input:
        f"{OUTPUT}/priority_list.csv",
        f"{OUTPUT}/pipeline_summary.json",
        f"{OUTPUT}/umap_plot.png",


# ── M01 ──────────────────────────────────────────────────────────────────────

rule pharokka_annotation:
    input:
        genomes = INPUT_GENOME,
    output:
        faa        = f"{INTERMEDIATE}/01_pharokka/all_proteins.faa",
        ffn        = f"{INTERMEDIATE}/01_pharokka/all_nucleotides.ffn",
        annot_tsv  = f"{INTERMEDIATE}/01_pharokka/annotation_table.tsv",
        nuc_lookup = f"{INTERMEDIATE}/01_pharokka/nucleotide_lookup.json",
    threads: config["paths"].get("pharokka_threads", 8)
    resources:
        mem_mb  = 16000,
        runtime = 480,
    log: f"{LOG_DIR}/m01_pharokka.log"
    conda: "workflow/envs/pharokka.yaml"
    script: "pipeline/modules/m01_pharokka.py"


# ── M02 ──────────────────────────────────────────────────────────────────────

rule lysis_module_identification:
    input:
        faa        = rules.pharokka_annotation.output.faa,
        ffn        = rules.pharokka_annotation.output.ffn,
        annot_tsv  = rules.pharokka_annotation.output.annot_tsv,
        nuc_lookup = rules.pharokka_annotation.output.nuc_lookup,
    output:
        candidates = f"{INTERMEDIATE}/02_lysis_modules/candidates.json",
        modules    = f"{INTERMEDIATE}/02_lysis_modules/modules.json",
        faa_out    = f"{INTERMEDIATE}/02_lysis_modules/candidates.faa",
        hmmer_hits = f"{INTERMEDIATE}/02_lysis_modules/hmmer_hits.tsv",
    threads: config["candidate_extraction"].get("hmmer_threads", 4)
    resources:
        mem_mb  = 8000,
        runtime = 120,
    log: f"{LOG_DIR}/m02_lysis_modules.log"
    conda: "workflow/envs/hmmer.yaml"
    script: "pipeline/modules/m02_lysis_modules.py"


# ── M02b — ESM-2 reference similarity (Stage 3 identification) ──────────────

rule reference_similarity:
    input:
        faa        = rules.pharokka_annotation.output.faa,
        candidates = rules.lysis_module_identification.output.candidates,
        ref_check  = ancient(expand(
            "{ref_dir}/{cls}_reference.npy",
            ref_dir = config.get("reference_db", "references"),
            cls     = ["endolysin", "holin", "spanin"],
        )) if Path(config.get("reference_db", "references")).exists() else [],
    output:
        scores = f"{INTERMEDIATE}/02_lysis_modules/ref_similarity_scores.tsv",
    resources:
        mem_mb  = 16000,
        runtime = 60,
    log: f"{LOG_DIR}/m02b_reference_similarity.log"
    script: "pipeline/modules/m02b_reference_similarity.py"


# ── M03 ──────────────────────────────────────────────────────────────────────

rule gate1_expressibility:
    input:
        candidates   = rules.lysis_module_identification.output.candidates,
        ref_done     = rules.reference_similarity.output.scores,
    output:
        passing = f"{INTERMEDIATE}/03_gate1/candidates_passing.json",
        results = f"{INTERMEDIATE}/03_gate1/gate1_results.tsv",
    resources:
        mem_mb  = 4000,
        runtime = 30,
    log: f"{LOG_DIR}/m03_gate1.log"
    conda: "workflow/envs/ml.yaml"
    script: "pipeline/modules/m03_gate1.py"


# ── M04 ──────────────────────────────────────────────────────────────────────

rule esm2_embeddings:
    input:
        candidates = rules.gate1_expressibility.output.passing,
    output:
        matrix   = f"{INTERMEDIATE}/04_embeddings/embedding_matrix.npy",
        index    = f"{INTERMEDIATE}/04_embeddings/embedding_index.json",
        manifest = f"{INTERMEDIATE}/04_embeddings/manifest.json",
    resources:
        mem_mb  = 32000,
        runtime = 360,
        gpu     = 1 if config["api"].get("esm2_backend") == "local" else 0,
    log: f"{LOG_DIR}/m04_embeddings.log"
    conda: "workflow/envs/ml.yaml"
    script: "pipeline/modules/m04_embeddings.py"


# ── M05 ──────────────────────────────────────────────────────────────────────

rule clustering:
    input:
        candidates = rules.gate1_expressibility.output.passing,
        matrix     = rules.esm2_embeddings.output.matrix,
        index      = rules.esm2_embeddings.output.index,
    output:
        summary = f"{INTERMEDIATE}/05_clusters/cluster_summary.json",
    resources:
        mem_mb  = 16000,
        runtime = 60,
    log: f"{LOG_DIR}/m05_clustering.log"
    conda: "workflow/envs/ml.yaml"
    script: "pipeline/modules/m05_clustering.py"


# ── M06 — conditional on pg_matching.enabled ─────────────────────────────────

if PG_ENABLED:
    rule pg_matching:
        input:
            candidates   = rules.gate1_expressibility.output.passing,
            # Serialized after clustering — both modules mutate candidates_passing.json
            # so they must NOT run in parallel
            cluster_done = rules.clustering.output.summary,
            pathogen_db  = config.get("targets", {}).get(
                "pathogen_db", "targets/pathogen_db.yaml"
            ),
        output:
            scores = f"{INTERMEDIATE}/06_pg_matching/pg_scores.tsv",
        resources:
            mem_mb  = 2000,
            runtime = 10,
        log: f"{LOG_DIR}/m06_pg_matching.log"
        conda: "workflow/envs/ml.yaml"
        script: "pipeline/modules/m06_pg_matching.py"

    PG_INPUT = f"{INTERMEDIATE}/06_pg_matching/pg_scores.tsv"

else:
    rule pg_matching_skip:
        output: touch(f"{INTERMEDIATE}/06_pg_matching/.skip")
        run: pass

    PG_INPUT = []


# ── M07 ──────────────────────────────────────────────────────────────────────

rule redundancy_collapse:
    input:
        candidates   = rules.gate1_expressibility.output.passing,
        matrix       = rules.esm2_embeddings.output.matrix,
        index        = rules.esm2_embeddings.output.index,
        modules      = rules.lysis_module_identification.output.modules,
        # Must wait for clustering AND PG matching — both mutate candidates_passing.json
        cluster_done = rules.clustering.output.summary,
        pg_done      = PG_INPUT,   # empty list when PG disabled — safe
    output:
        gate3_results = f"{INTERMEDIATE}/07_redundancy/gate3_results.tsv",
    resources:
        mem_mb  = 16000,
        runtime = 30,
    log: f"{LOG_DIR}/m07_redundancy.log"
    conda: "workflow/envs/ml.yaml"
    script: "pipeline/modules/m07_redundancy.py"


# ── M08 ──────────────────────────────────────────────────────────────────────

rule selection:
    input:
        candidates = rules.gate1_expressibility.output.passing,
        modules    = rules.lysis_module_identification.output.modules,
        matrix     = rules.esm2_embeddings.output.matrix,
        index      = rules.esm2_embeddings.output.index,
        gate3_done = rules.redundancy_collapse.output.gate3_results,
        pg_done    = PG_INPUT,
    output:
        summary = f"{INTERMEDIATE}/08_selection/selection_summary.json",
        curve   = f"{INTERMEDIATE}/08_selection/diversity_curve.tsv",
    resources:
        mem_mb  = 8000,
        runtime = 30,
    log: f"{LOG_DIR}/m08_selection.log"
    conda: "workflow/envs/ml.yaml"
    script: "pipeline/modules/m08_selection.py"


# ── M09 ──────────────────────────────────────────────────────────────────────

rule report_generation:
    input:
        candidates     = rules.gate1_expressibility.output.passing,
        selection_done = rules.selection.output.summary,
        cluster_done   = rules.clustering.output.summary,
    output:
        priority_list  = f"{OUTPUT}/priority_list.csv",
        reserve_list   = f"{OUTPUT}/reserve_list.csv",
        eliminated_log = f"{OUTPUT}/eliminated_log.csv",
        summary        = f"{OUTPUT}/pipeline_summary.json",
        umap_plot      = f"{OUTPUT}/umap_plot.png",
    resources:
        mem_mb  = 4000,
        runtime = 15,
    log: f"{LOG_DIR}/m09_report.log"
    conda: "workflow/envs/ml.yaml"
    script: "pipeline/modules/m09_report.py"


# ── M10 — active learning training (manual trigger) ──────────────────────────

rule active_learning_train:
    input:
        results = config.get("wetlab_results", "data/wetlab/round1.csv"),
        matrix  = rules.esm2_embeddings.output.matrix,
        index   = rules.esm2_embeddings.output.index,
    output:
        models_dir = directory("data/active_learning/models"),
        metadata   = "data/active_learning/model_metadata.json",
    resources:
        mem_mb  = 8000,
        runtime = 30,
    log: f"{LOG_DIR}/m10_active_learning.log"
    run:
        from pipeline.utils.logging_config import setup_logging
        from pipeline.active_learning.train import train
        setup_logging(config["paths"].get("log_dir", "logs"))
        train(
            results_csv = str(input.results),
            config_path = "config/config.yaml",
            output_dir  = "data/active_learning",
            append      = config.get("append_wetlab", False),
        )


# ── M11 — round 2 re-ranking (manual trigger) ────────────────────────────────

rule round2_selection:
    input:
        models   = rules.active_learning_train.output.models_dir,
        metadata = rules.active_learning_train.output.metadata,
        matrix   = rules.esm2_embeddings.output.matrix,
    output:
        reranked = "data/active_learning/round2_candidates.csv",
    resources:
        mem_mb  = 4000,
        runtime = 15,
    log: f"{LOG_DIR}/m11_round2_selection.log"
    run:
        from pipeline.utils.logging_config import setup_logging
        from pipeline.active_learning.predict import predict_and_rerank
        setup_logging(config["paths"].get("log_dir", "logs"))
        predict_and_rerank(
            config_path = "config/config.yaml",
            al_dir      = "data/active_learning",
            output_dir  = "data/active_learning",
        )


# ── Utility rules ─────────────────────────────────────────────────────────────

rule clean_intermediate:
    """Remove intermediate files, keep final outputs."""
    run:
        import shutil
        p = Path(INTERMEDIATE)
        if p.exists():
            shutil.rmtree(p)
            print(f"Removed: {p}")

rule clean_all:
    """Remove all pipeline data."""
    run:
        import shutil
        for d in [INTERMEDIATE, OUTPUT, "data/active_learning"]:
            p = Path(d)
            if p.exists():
                shutil.rmtree(p)
                print(f"Removed: {p}")

rule dag_png:
    """Export DAG as PNG. Requires graphviz."""
    shell:
        "snakemake --configfile config/config.yaml --dag | dot -Tpng > pipeline_dag.png"
        " && echo 'DAG saved: pipeline_dag.png'"
