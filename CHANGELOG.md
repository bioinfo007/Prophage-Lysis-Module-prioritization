# Changelog — prophage_lysis

All notable changes to this project are documented here.
Format: [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2025

### Complete rewrite from scratch

**Architecture**
- Replaced monolithic orchestrator with Snakemake DAG (full checkpointing, parallelism, HPC)
- Added Typer CLI (`run`, `resume`, `check`, `add-target`, `active-learning`, `round2-select`, `report-status`)
- Added standalone sequential runner (`run_pipeline.py`) for single-machine use without Snakemake
- Package installable via `pip install -e .` or `conda env create -f environment.yaml`

**Data model**
- Three parallel track dataclasses: `EndolysínRecord`, `HolinRecord`, `SpanínRecord`
- `LysisModule` links all three track records from one prophage locus
- `n_tm_helices` moved to `_BaseRecord` so all three tracks serialize it correctly
- All fields survive JSON round-trip via `to_dict()` / `from_dict()`

**M01 — Pharokka annotation**
- Nucleotide CDS sequences now extracted from GenBank output alongside proteins
- Nucleotide lookup JSON written for CAI calculation in M03

**M02 — Lysis module identification**
- Three parallel identification tracks with track-specific logic
- SAR endolysin detection: single N-terminal TM helix + downstream catalytic domain
- Genomic proximity module linkage (10 ORF window) connecting holin + endolysin + spanin
- Spanin i/o type inference from domain annotation, topology, and keyword

**M03 — Gate 1 expressibility filter**
- Track-aware thresholds: holins, endolysins, and spanins each have appropriate criteria
- Holins required to have TM helices (not penalized for them)
- SAR endolysins exempt from TM helix and GRAVY penalties
- CAI computed from actual nucleotide CDS sequences (not approximate amino acid-based)
- Eliminated candidates carry full audit trail (gate, reason, all flags)

**M04 — ESM-2 embeddings**
- Gate 1 runs before embedding — saves 30–50% compute
- Local ESM-2 (650M, grouped by length, batched, GPU auto-detected) or Atlas API
- Manifest JSON for O(1) resume without filesystem scan
- Zero-vector candidates excluded from matrix rather than silently injected

**M05 — Clustering**
- Per-track UMAP+HDBSCAN — endolysins and holins never share clusters
- UMAP 50D for HDBSCAN, spectral-initialized 2D for visualization
- Noise points assigned to nearest cluster centroid (not abandoned)

**M06 — PG chemistry matching**
- Optional (flag-controlled)
- Extensible via YAML + CLI `add-target` subcommand — no source code changes needed
- Hardcoded compatibility table: 5 catalytic domain types × 4 PG chemotypes
- M06 serialized after M05 in DAG to prevent race condition on `candidates_passing.json`

**M07 — Redundancy collapse**
- Block cosine similarity for N > 5000 (avoids OOM on large candidate pools)
- Composite representative criterion: module_complete bonus + CAI + fewest flags + lowest MW
- Full similarity edge list written for downstream analysis

**M08 — MaxMin diversity selection**
- Numba JIT-compiled MaxMin kernel (parallel CPU or CUDA GPU, same code path)
- Saturation detection: stops when marginal distance < threshold × initial distance
- Module co-selection: selecting an endolysin auto-selects its cognate holin + spanin
- Pathogen coverage extension: adds reserve candidates targeting under-served pathogens
- BLAST novelty check against SwissProt — fixed polling: now correctly checks `Status=READY` in response body (not `status_code == 200`)

**M09 — Report generation**
- `priority_list.csv`, `reserve_list.csv`, `eliminated_log.csv`, `pipeline_summary.json`
- Per-endolysin markdown reports with domain architecture, physicochemical properties, novelty
- 3-panel UMAP plot (one panel per track)

**M10/M11 — Active learning**
- Per-pathogen LogisticRegression (< 20 samples) or RandomForest (≥ 20) on ESM-2 features
- Multi-round append mode — combines wet lab data from multiple rounds
- Composite re-rank score = mean predicted probability across all target pathogens
- `round2_candidates.csv` sorted by composite score descending

**Utils**
- `numba_kernels.py`: GPU auto-detect, block cosine similarity, MaxMin, mean pairwise distance
- `pg_database.py`: YAML-backed extensible pathogen database
- `cai.py`: real E. coli relative adaptiveness table (Sharp & Li 1987)
- `hmmer.py`: corrected domain sets (PF07411, PF01471 removed from endolysin catalytic)

**Testing**
- 4 unit test files (data model, HMMER, utils, Gate 1)
- Integration tests using synthetic data — no databases required
- Fixture FASTA + annotation + nucleotide lookup for M02 logic tests
- GitHub Actions CI: lint, unit tests, Snakemake dry-run, integration tests
- SLURM cluster profile for HPC submission

**Fixed bugs from v1.x**
- `PF01471` (PG_binding_1) incorrectly in catalytic domain set → moved to CBD_DOMAINS
- `PF07411` (L,D-transpeptidase) incorrectly in catalytic domain set → removed
- CAI computed from amino acid frequencies (broken) → now from nucleotide CDS
- BLAST polling condition `status_code == 200` always true → now correctly reads `Status=READY` from body
- `n_tm_helices` not in `_BaseRecord` → lost on JSON serialization → fixed
- Saturation bug: could append same candidate twice → fixed with `remaining` mask
- HDBSCAN noise points abandoned → now assigned to nearest centroid
- M05/M06/M07 parallel DAG mutation of `candidates_passing.json` → serialized in Snakefile

---

## [1.x] — 2024

Initial prototype. Monolithic orchestrator, broken CAI, incorrect domain sets.
Not documented in this changelog.
