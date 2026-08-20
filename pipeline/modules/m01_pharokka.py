"""
m01_pharokka.py
===============
Module 01: Run Pharokka annotation on phage genome FASTA files.

Input:  data/input/phage_genomes/*.fasta|fa|fna|ffn
Output: data/intermediate/01_pharokka/{genome_id}/
        data/intermediate/01_pharokka/all_proteins.faa
        data/intermediate/01_pharokka/all_nucleotides.ffn   ← new: CDS nucleotide seqs
        data/intermediate/01_pharokka/annotation_table.tsv

Key change from original: nucleotide CDS sequences are extracted alongside
protein sequences. This enables real CAI calculation in M03 (Gate 1).
"""

import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

log = logging.getLogger("m01_pharokka")


def run(cfg: dict) -> None:
    genome_dir  = Path(cfg["paths"]["genome_input_dir"])
    pharokka_db = cfg["paths"]["pharokka_db"]
    threads     = cfg["paths"].get("pharokka_threads", 8)
    out_root    = Path(cfg["paths"]["intermediate_dir"]) / "01_pharokka"
    out_root.mkdir(parents=True, exist_ok=True)

    genome_files = sorted(
        list(genome_dir.glob("*.fasta")) +
        list(genome_dir.glob("*.fa"))    +
        list(genome_dir.glob("*.fna"))   +
        list(genome_dir.glob("*.ffn"))
    )

    if not genome_files:
        raise FileNotFoundError(f"No FASTA files in {genome_dir}")

    log.info(f"Annotating {len(genome_files)} genome(s) with Pharokka")

    all_proteins:     List[Dict] = []
    all_nucleotides:  List[Dict] = []
    annotation_rows:  List[Dict] = []

    for genome_path in genome_files:
        genome_id  = genome_path.stem
        genome_out = out_root / genome_id
        genome_out.mkdir(exist_ok=True)

        log.info(f"  → {genome_id}")

        _run_pharokka(genome_path, genome_out, pharokka_db, threads, genome_id)

        proteins, nucleotides = _parse_proteins_and_cds(genome_out, genome_id)
        annots                = _parse_annotations(genome_out, genome_id)

        all_proteins.extend(proteins)
        all_nucleotides.extend(nucleotides)
        annotation_rows.extend(annots)

        log.info(
            f"    {genome_id}: {len(proteins)} proteins, "
            f"{len(nucleotides)} CDS nucleotide sequences"
        )

    # Write combined protein FASTA
    combined_faa = out_root / "all_proteins.faa"
    with open(combined_faa, "w") as f:
        for p in all_proteins:
            f.write(f">{p['header']}\n{p['sequence']}\n")

    # Write combined nucleotide FASTA
    combined_ffn = out_root / "all_nucleotides.ffn"
    with open(combined_ffn, "w") as f:
        for n in all_nucleotides:
            f.write(f">{n['candidate_id']}\n{n['nucleotide_seq']}\n")

    # Write nucleotide lookup JSON (candidate_id → nucleotide_seq)
    nuc_lookup = {n["candidate_id"]: n["nucleotide_seq"] for n in all_nucleotides}
    (out_root / "nucleotide_lookup.json").write_text(
        json.dumps(nuc_lookup, indent=2)
    )

    # Write annotation table
    annot_tsv = out_root / "annotation_table.tsv"
    if annotation_rows:
        with open(annot_tsv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=annotation_rows[0].keys(),
                delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(annotation_rows)

    log.info(
        f"M01 complete — {len(all_proteins)} proteins across "
        f"{len(genome_files)} genomes"
    )


# ── Pharokka runner ───────────────────────────────────────────────────────────

def _run_pharokka(
    genome_path: Path,
    out_dir:     Path,
    db_dir:      str,
    threads:     int,
    prefix:      str,
) -> None:
    # Skip if already done (checkpoint)
    if (out_dir / f"{prefix}.faa").exists():
        log.info(f"    Pharokka output exists for {prefix} — skipping")
        return

    cmd = [
        "pharokka.py",
        "-i", str(genome_path),
        "-o", str(out_dir),
        "-d", db_dir,
        "-t", str(threads),
        "-p", prefix,
        "-f",   # force overwrite
    ]

    log.debug(f"    cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        log.error(f"Pharokka stderr:\n{result.stderr[-3000:]}")
        raise RuntimeError(
            f"Pharokka failed for {genome_path} (exit {result.returncode})"
        )


# ── Protein + CDS extraction ──────────────────────────────────────────────────

def _parse_proteins_and_cds(
    genome_out: Path,
    genome_id:  str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse Pharokka .faa (proteins) and extract nucleotide CDS sequences
    from the GenBank output (most reliable source for nucleotide sequences).

    Returns (proteins, nucleotides) where each is a list of dicts.
    """
    proteins:    List[Dict] = []
    nucleotides: List[Dict] = []

    # Build nucleotide lookup from GBK
    nuc_from_gbk: Dict[str, str] = {}
    gbk_files = list(genome_out.glob("*.gbk"))
    if gbk_files:
        nuc_from_gbk = _extract_cds_from_gbk(gbk_files[0])

    # Parse protein FASTA
    faa_files = list(genome_out.glob("*.faa"))
    if not faa_files:
        log.warning(f"No .faa file in {genome_out}")
        return proteins, nucleotides

    for record in SeqIO.parse(faa_files[0], "fasta"):
        parts    = record.description.split(None, 1)
        locus    = parts[0]
        function = parts[1] if len(parts) > 1 else "hypothetical protein"

        candidate_id = f"{genome_id}__{locus}"
        header       = f"{candidate_id} {function} genome={genome_id}"

        proteins.append({
            "candidate_id": candidate_id,
            "genome_id":    genome_id,
            "protein_id":   locus,
            "header":       header,
            "sequence":     str(record.seq).rstrip("*"),
            "function":     function,
        })

        nuc_seq = nuc_from_gbk.get(locus, "")
        nucleotides.append({
            "candidate_id":  candidate_id,
            "nucleotide_seq": nuc_seq,
        })

    return proteins, nucleotides


def _extract_cds_from_gbk(gbk_path: Path) -> Dict[str, str]:
    """
    Extract locus_tag → nucleotide CDS sequence from GenBank file.
    This is the ground-truth CDS nucleotide sequence for CAI calculation.
    """
    nuc: Dict[str, str] = {}

    for record in SeqIO.parse(gbk_path, "genbank"):
        genome_seq = record.seq
        for feature in record.features:
            if feature.type != "CDS":
                continue
            locus = feature.qualifiers.get("locus_tag", [""])[0]
            if not locus:
                continue
            try:
                cds_seq = str(feature.extract(genome_seq))
                nuc[locus] = cds_seq
            except Exception as e:
                log.debug(f"CDS extraction failed for {locus}: {e}")

    return nuc


# ── Annotation parsing ────────────────────────────────────────────────────────

def _parse_annotations(genome_out: Path, genome_id: str) -> List[Dict]:
    """
    Parse Pharokka *_cds_final_merged_output.tsv for per-protein annotations.
    Falls back to GBK parsing if TSV is missing.
    """
    annotations: List[Dict] = []

    # Primary: merged TSV
    merged_files = list(genome_out.glob("*_cds_final_merged_output.tsv"))
    if merged_files:
        try:
            with open(merged_files[0]) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    gene_id = row.get("gene", "").strip()
                    if not gene_id:
                        continue
                    annotations.append({
                        "candidate_id":  f"{genome_id}__{gene_id}",
                        "genome_id":     genome_id,
                        "protein_id":    gene_id,
                        "function":      row.get("annot", "hypothetical protein").strip(),
                        "category":      row.get("category", "unknown function").strip(),
                        "start":         _safe_int(row.get("start")),
                        "end":           _safe_int(row.get("stop")),
                        "strand":        row.get("strand", ""),
                        "organism":      row.get("organism", genome_id),
                    })
            log.debug(
                f"    Parsed {len(annotations)} annotations from "
                f"{merged_files[0].name}"
            )
            return annotations
        except Exception as e:
            log.warning(f"Could not parse merged TSV: {e} — falling back to GBK")

    # Fallback: GBK
    gbk_files = list(genome_out.glob("*.gbk"))
    if gbk_files:
        for record in SeqIO.parse(gbk_files[0], "genbank"):
            for feature in record.features:
                if feature.type != "CDS":
                    continue
                locus    = feature.qualifiers.get("locus_tag", ["unknown"])[0]
                function = feature.qualifiers.get("product",   ["hypothetical protein"])[0]
                annotations.append({
                    "candidate_id":  f"{genome_id}__{locus}",
                    "genome_id":     genome_id,
                    "protein_id":    locus,
                    "function":      function,
                    "category":      "",
                    "start":         int(feature.location.start),
                    "end":           int(feature.location.end),
                    "strand":        feature.location.strand,
                    "organism":      genome_id,
                })

    return annotations


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
