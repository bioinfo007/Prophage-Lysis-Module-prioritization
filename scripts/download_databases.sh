#!/usr/bin/env bash
# scripts/download_databases.sh
# =============================================================================
# Download all required databases for the prophage_lysis pipeline.
#
# Usage:
#   bash scripts/download_databases.sh [--dest /path/to/databases]
#   bash scripts/download_databases.sh --pharokka-only
#   bash scripts/download_databases.sh --pfam-only
#
# Requirements: wget, gunzip, hmmpress, conda (for pharokka)
# Disk space:   ~4 GB total
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DEST="${1:-/data/prophage_lysis_dbs}"
PHAROKKA_ONLY=0
PFAM_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dest)           DEST="$2";          shift 2 ;;
        --pharokka-only)  PHAROKKA_ONLY=1;    shift ;;
        --pfam-only)      PFAM_ONLY=1;        shift ;;
        --help|-h)
            echo "Usage: $0 [--dest DIR] [--pharokka-only] [--pfam-only]"
            exit 0
            ;;
        *)  shift ;;
    esac
done

mkdir -p "$DEST"
echo "=== prophage_lysis database download ==="
echo "Destination: $DEST"
echo ""

# ── Pfam-A HMM database ───────────────────────────────────────────────────────

download_pfam() {
    PFAM_DIR="$DEST/pfam"
    mkdir -p "$PFAM_DIR"

    PFAM_URL="https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"

    if [[ -f "$PFAM_DIR/Pfam-A.hmm.h3i" ]]; then
        echo "[SKIP] Pfam-A.hmm already downloaded and indexed"
        return 0
    fi

    echo "[1/2] Downloading Pfam-A.hmm (~350 MB compressed)..."
    wget -q --show-progress -O "$PFAM_DIR/Pfam-A.hmm.gz" "$PFAM_URL"

    echo "[2/2] Decompressing..."
    gunzip -f "$PFAM_DIR/Pfam-A.hmm.gz"

    echo "      Indexing with hmmpress..."
    hmmpress "$PFAM_DIR/Pfam-A.hmm"

    echo "[OK]  Pfam-A.hmm ready: $PFAM_DIR/Pfam-A.hmm"
    echo "      Update config.yaml:"
    echo "        pfam_hmm: \"$PFAM_DIR/Pfam-A.hmm\""
}

# ── Pharokka databases ────────────────────────────────────────────────────────

download_pharokka() {
    PHAROKKA_DIR="$DEST/pharokka_db"

    if [[ -d "$PHAROKKA_DIR" ]] && [[ -n "$(ls -A "$PHAROKKA_DIR" 2>/dev/null)" ]]; then
        echo "[SKIP] Pharokka database already exists: $PHAROKKA_DIR"
        return 0
    fi

    mkdir -p "$PHAROKKA_DIR"

    echo "[1/1] Downloading Pharokka databases (~2.5 GB)..."
    echo "      This uses Pharokka's official downloader."

    if ! command -v install-database.py &>/dev/null; then
        echo "[ERROR] install-database.py not found."
        echo "        Make sure the conda environment is activated:"
        echo "          conda activate prophage_lysis"
        exit 1
    fi

    install-database.py -o "$PHAROKKA_DIR"

    echo "[OK]  Pharokka databases ready: $PHAROKKA_DIR"
    echo "      Update config.yaml:"
    echo "        pharokka_db: \"$PHAROKKA_DIR\""
}

# ── Run ───────────────────────────────────────────────────────────────────────

if [[ $PFAM_ONLY -eq 1 ]]; then
    download_pfam
elif [[ $PHAROKKA_ONLY -eq 1 ]]; then
    download_pharokka
else
    download_pfam
    echo ""
    download_pharokka
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Download complete ==="
echo ""
echo "Add these paths to your config.yaml:"

if [[ $PFAM_ONLY -eq 0 ]]; then
    echo "  pharokka_db: \"$DEST/pharokka_db\""
fi
if [[ $PHAROKKA_ONLY -eq 0 ]]; then
    echo "  pfam_hmm:    \"$DEST/pfam/Pfam-A.hmm\""
fi

echo ""
echo "Then verify everything is ready:"
echo "  prophage_lysis check --config config/config.yaml"
