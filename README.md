# prophage_lysis v2.0.0

**Phenotype-Guided and AI-Driven Discovery of Prophage Lysis Modules from Marine-Derived *Vibrio* Species**

A Snakemake pipeline for mining endolysins, holins, and spanins from phage genome FASTAs using ESM-2 protein language model embeddings, UMAP+HDBSCAN clustering, and MaxMin diversity selection, with an active learning loop that re-ranks reserve candidates from wet lab activity data.

---

## Pipeline overview

```
Input: phage genome FASTAs (any source — PHASTER, manual extraction, etc.)
          ↓
M01  Pharokka annotation
M02  Lysis module identification  (3 tracks: endolysin | holin | spanin)
M03  Gate 1 — expressibility filter (track-aware, real CAI from nucleotide CDS)
M04  ESM-2 embeddings (survivors only — 30–50% compute saving)
M05  UMAP + HDBSCAN clustering (per track, noise → nearest centroid)
M06  PG chemistry matching [optional flag]
M07  Redundancy collapse (per track, block similarity for N > 5000)
M08  Module-aware MaxMin diversity selection (Numba JIT)
M09  Report — priority_list.csv, UMAP plot, per-candidate markdown
          ↓
     W5: Expression + activity testing (wet lab)
          ↓
M10  Active learning — classifier training per pathogen (ESM-2 features)
M11  Re-ranked reserve list for round 2
          ↓
     W6: Round 2 expression  →  loop until coverage satisfied
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-lab/prophage_lysis.git
cd prophage_lysis

# Create conda environment (installs all dependencies)
conda env create -f environment.yaml
conda activate prophage_lysis

# Install package in editable mode
pip install -e .

# Verify install
prophage_lysis --help
```

### GPU acceleration (optional)

If a CUDA GPU is available, replace the PyTorch line in `environment.yaml`:
```yaml
# Replace:
- cpuonly
# With (for CUDA 11.8):
- pytorch-cuda=11.8
```
Then reinstall. ESM-2 embeddings and cosine similarity kernels will use the GPU automatically.

---

## Setup

### 1. Download databases

```bash
# Pharokka database
install-database.py -o /path/to/pharokka_db

# Pfam-A HMM database
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
gunzip Pfam-A.hmm.gz
hmmpress Pfam-A.hmm
```

### 2. Edit config

```bash
cp config/config.yaml config/my_run.yaml
# Edit paths: pharokka_db, pfam_hmm, genome_input_dir
```

### 3. Add phage genomes

```bash
cp /path/to/my_phage_genomes/*.fasta data/input/phage_genomes/
```

---

## Running

```bash
# Check everything is set up correctly
prophage_lysis check --config config/my_run.yaml

# Dry run — see the DAG without executing
prophage_lysis run --config config/my_run.yaml --dry-run

# Full run (8 cores, recommended)
prophage_lysis run --config config/my_run.yaml --cores 8

# Resume from a specific module after failure
prophage_lysis resume --from-module 5 --config config/my_run.yaml

# Check current progress
prophage_lysis report-status --config config/my_run.yaml
```

---

## Optional features

### Enable PG chemistry matching

Edit `config.yaml`:
```yaml
pg_matching:
  enabled: true
  enforce_coverage: true
```

View default target pathogens:
```bash
cat targets/pathogen_db.yaml
```

Add a new target pathogen:
```bash
prophage_lysis add-target \
  --id "vibrio_vulnificus" \
  --name "Vibrio vulnificus" \
  --species "Vibrio vulnificus" \
  --gram-stain negative \
  --pg-chemotype gram_negative_dap_om_barrier \
  --aquaculture-host "oyster, sea bream"
```

### ESM-2 local mode (faster, no internet required)

Edit `config.yaml`:
```yaml
api:
  esm2_backend: local
  local_batch_size: 16
```

First run downloads the ESM-2 650M model (~1.5 GB). GPU is used automatically if available.

---

## Active learning loop

After wet lab expression and activity testing (W5):

```bash
# Train classifiers from wet lab results
prophage_lysis active-learning \
  --results data/wetlab/round1.csv \
  --config config/my_run.yaml

# Re-rank reserve candidates for round 2
prophage_lysis round2-select --config config/my_run.yaml
```

For round 2 results, use `--append` to combine with round 1 data:
```bash
prophage_lysis active-learning \
  --results data/wetlab/round2.csv \
  --config config/my_run.yaml \
  --append
```

Wet lab CSV format:
```
candidate_id,pathogen_id,mic_ug_ml,kill_pct_6h,active
genome1__endo01,vibrio_harveyi,12.5,85.3,1
genome1__endo02,vibrio_harveyi,>200,8.1,0
```

---

## Outputs

| File | Description |
|---|---|
| `data/output/priority_list.csv` | Selected module components ranked by endolysin diversity rank |
| `data/output/reserve_list.csv` | Backup candidates for round 2 |
| `data/output/eliminated_log.csv` | Full audit trail with elimination reason per candidate |
| `data/output/pipeline_summary.json` | Run statistics |
| `data/output/candidate_reports/*.md` | Per-endolysin markdown with domain architecture, physicochemical properties, novelty status |
| `data/output/umap_plot.png` | 3-panel UMAP visualization (endolysin, holin, spanin) |
| `data/active_learning/round2_candidates.csv` | Re-ranked reserve list after active learning |

---

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pipeline --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

---

## Architecture decisions

| Decision | Rationale |
|---|---|
| Snakemake workflow | Industry standard for reproducible bioinformatics; native HPC support via `--cluster` |
| ESM-2 650M | Best public protein LM at this scale; sequence-agnostic functional representations |
| Gate 1 before ESM-2 | Saves 30–50% embedding compute by filtering non-expressible candidates first |
| Per-track clustering | Endolysins and holins should not share clusters — different functional space |
| Numba JIT kernels | C-speed similarity + MaxMin without leaving Python |
| Module co-selection | Selecting endolysins automatically co-selects cognate holin+spanin for combinatorial expression |
| SAR endolysin correction | SAR endolysins have 1 N-terminal TM helix — misclassified as holins without this fix |
| Block cosine similarity | Avoids OOM for N > 5000 by computing only edges above threshold |
| BLAST novelty check | Tags candidates not in SwissProt — novel enzymes are highest-value outputs |

---

## Citation

If you use this pipeline, please cite:

> [Authors]. Phenotype-guided and AI-driven discovery of prophage lysis modules from marine *Vibrio* species for aquaculture biocontrol. *[Journal]*, 2025.

---

## License

MIT License. See `LICENSE`.

## Contact

Pukyong National University, Department of Marine Biotechnology, Busan, Korea.
