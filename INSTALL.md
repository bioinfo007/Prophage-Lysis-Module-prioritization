# Installation Guide — prophage_lysis v2.0.0

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Linux | Ubuntu 20.04+ / RHEL 8+ | macOS works but not tested |
| conda / mamba | any | miniconda recommended |
| git | any | for cloning |
| graphviz | optional | for DAG visualization |

---

## Step 1: Clone and install

```bash
git clone https://github.com/your-lab/prophage_lysis.git
cd prophage_lysis

# Create and activate environment (~5 min first time)
conda env create -f environment.yaml
conda activate prophage_lysis

# Install pipeline as editable package
pip install -e .

# Confirm CLI works
prophage_lysis --help
```

> **PyTorch is optional.** The default backend (`atlas`) uses a REST API — no torch needed.
> For the `local` ESM-2 backend (offline inference), install torch after activating:
> ```bash
> # CPU — works on any machine, no GPU required:
> pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
> pip install fair-esm
>
> # GPU — CUDA 12.1:
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install fair-esm
> ```
> Then set `api: esm2_backend: local` in `config.yaml`.
> Do NOT use plain `pip install torch` — it pulls ~2 GB of CUDA packages
> regardless of whether you have a GPU.

---

## Step 2: Download databases

### Pharokka database

```bash
# ~2 GB download
install-database.py -o /data/pharokka_db
```

### Pfam-A HMM database

```bash
# Download from EBI (~300 MB compressed)
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz -P /data/
gunzip /data/Pfam-A.hmm.gz

# Index for HMMER (required before use)
hmmpress /data/Pfam-A.hmm
# Creates: Pfam-A.hmm.h3f  Pfam-A.hmm.h3i  Pfam-A.hmm.h3m  Pfam-A.hmm.h3p
```

---

## Step 3: Configure

```bash
cp config/config.yaml config/my_run.yaml
```

Edit `config/my_run.yaml`:

```yaml
paths:
  genome_input_dir: "data/input/phage_genomes"  # your genome FASTAs go here
  pharokka_db:      "/data/pharokka_db"          # path from Step 2
  pfam_hmm:         "/data/Pfam-A.hmm"           # path from Step 2
  intermediate_dir: "data/intermediate"
  output_dir:       "data/output"
  pharokka_threads: 8    # adjust to your server
```

---

## Step 4: Add phage genomes

Copy your phage genome FASTA files to the input directory:

```bash
cp /path/to/your/phages/*.fasta data/input/phage_genomes/
ls data/input/phage_genomes/
```

Accepted formats: `.fasta`, `.fa`, `.fna`, `.ffn`
The pipeline is source-agnostic — genomes can come from PHASTER, manual extraction,
prophage databases (PhagesDB, INPHARED), or direct sequencing.

---

## Step 5: Check setup

```bash
prophage_lysis check --config config/my_run.yaml
```

Expected output:
```
✓  genome_input_dir: data/input/phage_genomes (12 genomes)
✓  pfam_hmm: /data/Pfam-A.hmm
✓  pharokka_db: /data/pharokka_db
✓  Pfam-A.hmm hmmpress index: OK
[OK] All checks passed — ready to run.
```

---

## Step 6: Run

### Option A: Snakemake (recommended — handles checkpointing, parallelism, HPC)

```bash
# Dry run first — prints what will execute
prophage_lysis run --config config/my_run.yaml --dry-run

# Full run
prophage_lysis run --config config/my_run.yaml --cores 8
```

### Option B: Standalone sequential runner (simpler, no Snakemake)

```bash
# Full run
python run_pipeline.py --config config/my_run.yaml

# Resume from module 4 after a failure
python run_pipeline.py --config config/my_run.yaml --from 4

# Run only one module
python run_pipeline.py --config config/my_run.yaml --only 5
```

---

## Step 7: Outputs

After a successful run:

```
data/output/
├── priority_list.csv         ← submit these for expression
├── reserve_list.csv          ← backup candidates
├── eliminated_log.csv        ← full audit trail
├── pipeline_summary.json     ← run statistics
├── umap_plot.png             ← UMAP visualization
├── candidate_reports/        ← per-endolysin markdown reports
│   ├── genome1__endo001.md
│   └── ...
└── logs/                     ← per-module logs
```

---

## Optional: GPU acceleration

If a CUDA GPU is available, replace the PyTorch install in `environment.yaml`:

```yaml
# Remove:
- cpuonly
# Add (for CUDA 11.8):
- pytorch-cuda=11.8
```

Then switch ESM-2 to local mode in `config.yaml`:
```yaml
api:
  esm2_backend: local
  local_batch_size: 32   # increase for GPU
```

First run downloads ESM-2 650M model (~1.5 GB). All embedding and similarity
kernels switch to GPU automatically — no code changes needed.

---

## Optional: PG chemistry matching

Enable in config:
```yaml
pg_matching:
  enabled: true
```

Add new target pathogens without editing code:
```bash
prophage_lysis add-target \
  --id "vibrio_vulnificus" \
  --name "Vibrio vulnificus" \
  --species "Vibrio vulnificus" \
  --gram-stain negative \
  --pg-chemotype gram_negative_dap_om_barrier \
  --aquaculture-host "oyster, flounder"
```

---

## Optional: HPC (SLURM)

```bash
snakemake --configfile config/my_run.yaml \
  --cores 200 --use-conda \
  --cluster "sbatch \
    --mem={resources.mem_mb}M \
    --cpus-per-task={threads} \
    --time={resources.runtime} \
    --output=data/output/logs/slurm_%j.out" \
  --jobs 50 \
  --latency-wait 60
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `hmmpress` not found | `conda install -c bioconda hmmer` |
| `pharokka.py` not found | `conda install -c bioconda pharokka` |
| ESM-2 API timeout | Switch to local backend or increase `request_timeout` |
| OOM in M07 | Reduce `gate3.similarity_threshold` or switch to block mode (auto for N>5000) |
| `conda activate` fails | Run `conda init bash && source ~/.bashrc` first |
| Snakemake DAG error | Run `prophage_lysis run -n` to see the DAG without executing |

---

## Testing

```bash
# Run unit tests (no databases required)
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=pipeline --cov-report=html
open htmlcov/index.html
```
