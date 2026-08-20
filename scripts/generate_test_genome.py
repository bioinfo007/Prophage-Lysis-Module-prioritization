"""
scripts/generate_test_genome.py
================================
Generate synthetic phage genome FASTAs containing known lysis module genes
for integration testing without requiring real phage sequences or databases.

Synthetic proteins embedded in genome:
  - 1 endolysin (CHAP domain — PF04851)
  - 1 holin (2 TM helices, short, hydrophobic)
  - 1 i-spanin (1 TM helix, Rz-like)
  - 10 structural/other proteins (noise)

The sequences are biologically plausible but randomised from amino acid
frequency distributions — not real proteins.

Usage:
  python scripts/generate_test_genome.py --n-genomes 3 --output data/input/phage_genomes/
  python scripts/generate_test_genome.py --seed 42 --n-endolysins 5
"""

from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

# Amino acid frequency table (E. coli proteome average)
_AA_FREQ = {
    'A': 9.69, 'R': 5.52, 'N': 3.87, 'D': 5.30, 'C': 1.28,
    'Q': 3.90, 'E': 6.32, 'G': 7.08, 'H': 2.28, 'I': 5.49,
    'L': 9.68, 'K': 4.97, 'M': 2.32, 'F': 3.88, 'P': 5.02,
    'S': 6.71, 'T': 5.34, 'W': 1.37, 'Y': 2.76, 'V': 6.73,
}
_AA_LIST  = list(_AA_FREQ.keys())
_AA_WEIGHTS = [_AA_FREQ[a] for a in _AA_LIST]

# Hydrophobic AAs for TM helices
_HYDROPHOBIC = list("LLLLVVVVIIIIFFFFWWWWA")


def _random_protein(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(_AA_LIST, weights=_AA_WEIGHTS, k=length))


def _random_codon(aa: str, rng: random.Random) -> str:
    """Return a random codon for the given amino acid (simplified table)."""
    codons = {
        'A': ['GCT','GCC','GCA','GCG'], 'R': ['CGT','CGC','CGA','CGG','AGA','AGG'],
        'N': ['AAT','AAC'], 'D': ['GAT','GAC'], 'C': ['TGT','TGC'],
        'Q': ['CAA','CAG'], 'E': ['GAA','GAG'], 'G': ['GGT','GGC','GGA','GGG'],
        'H': ['CAT','CAC'], 'I': ['ATT','ATC','ATA'], 'L': ['TTA','TTG','CTT','CTC','CTA','CTG'],
        'K': ['AAA','AAG'], 'M': ['ATG'], 'F': ['TTT','TTC'], 'P': ['CCT','CCC','CCA','CCG'],
        'S': ['TCT','TCC','TCA','TCG','AGT','AGC'], 'T': ['ACT','ACC','ACA','ACG'],
        'W': ['TGG'], 'Y': ['TAT','TAC'], 'V': ['GTT','GTC','GTA','GTG'],
        '*': ['TAA','TAG','TGA'],
    }
    return rng.choice(codons.get(aa, ['NNN']))


def _protein_to_cds(protein: str, rng: random.Random) -> str:
    """Back-translate protein to CDS nucleotide sequence."""
    codons = [_random_codon(aa, rng) for aa in protein]
    codons.append(_random_codon('*', rng))   # stop codon
    return "".join(codons)


# ── Synthetic lysis module proteins ──────────────────────────────────────────

def _make_endolysin(rng: random.Random, length: int = 160) -> tuple:
    """
    Synthetic endolysin: moderate size, hydrophilic, stable.
    No real CHAP domain — just a realistic endolysin-like sequence.
    """
    # Core: random hydrophilic protein
    protein = _random_protein(length, rng)
    # Ensure GRAVY < 0 (hydrophilic) by biasing toward charged/polar AAs
    hydrophilic_aa = list("KRKRDEDENNQQSSTTS")
    core = [rng.choices([rng.choice(hydrophilic_aa), rng.choice(_AA_LIST)],
                         weights=[0.4, 0.6])[0] for _ in range(length)]
    protein = "M" + "".join(core[1:])
    return protein, _protein_to_cds(protein, rng)


def _make_holin(rng: random.Random) -> tuple:
    """
    Synthetic holin: short, two TM helices.
    TM helices are runs of hydrophobic AAs.
    """
    n_term  = "MKK" + _random_protein(10, rng)                     # short hydrophilic N-term
    tm1     = "".join(rng.choices(_HYDROPHOBIC, k=20))             # TM helix 1 (20 aa)
    loop    = _random_protein(rng.randint(5, 15), rng)             # cytoplasmic loop
    tm2     = "".join(rng.choices(_HYDROPHOBIC, k=20))             # TM helix 2 (20 aa)
    c_term  = "KR" + _random_protein(rng.randint(5, 20), rng)     # charged C-terminus
    protein = n_term + tm1 + loop + tm2 + c_term
    return protein, _protein_to_cds(protein, rng)


def _make_ispanin(rng: random.Random) -> tuple:
    """
    Synthetic i-spanin: one N-terminal TM helix, periplasmic coiled-coil.
    """
    n_term  = "MK"
    tm      = "".join(rng.choices(_HYDROPHOBIC, k=20))             # single TM helix
    peri    = _random_protein(rng.randint(60, 100), rng)           # periplasmic domain
    protein = n_term + tm + peri
    return protein, _protein_to_cds(protein, rng)


def _make_ospanin(rng: random.Random) -> tuple:
    """
    Synthetic o-spanin: signal peptide + outer membrane lipoprotein.
    No TM helices.
    """
    # Signal peptide: +++ (n-region) + hydrophobic (h-region) + cleavage
    sig_pep = "MKK" + "".join(rng.choices(["L","V","I","A"], k=12)) + "C"
    body    = _random_protein(rng.randint(50, 100), rng)
    protein = sig_pep + body
    return protein, _protein_to_cds(protein, rng)


def _make_structural_protein(rng: random.Random) -> tuple:
    """Random structural protein (noise — should NOT be identified as lysis module)."""
    length  = rng.randint(150, 600)
    protein = "M" + _random_protein(length - 1, rng)
    return protein, _protein_to_cds(protein, rng)


# ── GenBank-like output ───────────────────────────────────────────────────────

def _write_gbk(genome_id: str, genes: list, out_path: Path) -> None:
    """
    Write a minimal GenBank file that Pharokka can parse.
    genes: list of (locus_tag, product, protein_seq, nuc_seq, start, end, strand)
    """
    genome_len = max(g[5] for g in genes) + 100

    lines = [
        f"LOCUS       {genome_id:<16} {genome_len} bp    DNA     linear   PHG",
        f"DEFINITION  Synthetic phage genome {genome_id} for pipeline testing.",
        f"ACCESSION   {genome_id}",
        f"VERSION     {genome_id}.1",
        "FEATURES             Location/Qualifiers",
        f'     source          1..{genome_len}',
        f'                     /organism="Synthetic phage"',
        f'                     /mol_type="genomic DNA"',
    ]

    for locus, product, protein, nuc, start, end, strand in genes:
        loc = f"{start}..{end}" if strand == "+" else f"complement({start}..{end})"
        lines += [
            f"     CDS             {loc}",
            f'                     /locus_tag="{locus}"',
            f'                     /product="{product}"',
            f'                     /translation="{protein}"',
        ]

    # Minimal ORIGIN section (N's — sequence doesn't matter for testing)
    lines.append("ORIGIN")
    genome_seq = "n" * genome_len
    for i in range(0, genome_len, 60):
        chunk = genome_seq[i:i+60]
        lines.append(f"{i+1:9d} {' '.join(chunk[j:j+10] for j in range(0, len(chunk), 10))}")
    lines.append("//")

    out_path.write_text("\n".join(lines) + "\n")


# ── FASTA output ──────────────────────────────────────────────────────────────

def _write_fasta(genome_id: str, genes: list, out_path: Path) -> None:
    """Write protein FASTA (simulates Pharokka .faa output)."""
    lines = []
    for locus, product, protein, nuc, start, end, strand in genes:
        cid = f"{genome_id}__{locus}"
        lines.append(f">{cid} {product}")
        lines.append(protein)
    out_path.write_text("\n".join(lines) + "\n")


def _write_annotations(genome_id: str, genes: list, out_path: Path) -> None:
    """Write annotation TSV (simulates Pharokka merged output)."""
    rows = ["candidate_id\tgenome_id\tprotein_id\tfunction\tcategory\tstart\tend\tstrand\torganism"]
    for locus, product, protein, nuc, start, end, strand in genes:
        cid = f"{genome_id}__{locus}"
        rows.append(
            f"{cid}\t{genome_id}\t{locus}\t{product}\t"
            f"lysis\t{start}\t{end}\t{strand}\t{genome_id}"
        )
    out_path.write_text("\n".join(rows) + "\n")


def _write_nucleotides(genome_id: str, genes: list, out_path: Path) -> None:
    """Write nucleotide CDS FASTA and lookup JSON."""
    import json
    lookup = {}
    fasta_lines = []
    for locus, product, protein, nuc, start, end, strand in genes:
        cid = f"{genome_id}__{locus}"
        lookup[cid] = nuc
        fasta_lines.append(f">{cid}")
        fasta_lines.append(nuc)
    out_path.write_text("\n".join(fasta_lines) + "\n")
    return lookup


# ── Main generator ────────────────────────────────────────────────────────────

def generate_genome(
    genome_id:      str,
    rng:            random.Random,
    n_endolysins:   int = 2,
    n_holins:       int = 2,
    n_spanins:      int = 1,
    n_noise:        int = 12,
) -> list:
    """
    Generate one synthetic phage genome gene list.
    Returns genes sorted by genomic position.
    """
    genes = []
    pos   = 100    # start position

    def _add(locus, product, protein, nuc):
        nonlocal pos
        start = pos
        end   = pos + len(nuc) - 4   # exclude stop codon from displayed range
        genes.append((locus, product, protein, nuc, start, end, "+"))
        pos   = end + rng.randint(10, 50)   # intergenic gap

    # Add lysis cluster: holin(s) → endolysin(s) → spanin(s)
    # (biologically: holin usually precedes endolysin in lambda-like order)
    for i in range(n_holins):
        protein, nuc = _make_holin(rng)
        _add(f"holin_{i+1:03d}", "phage holin", protein, nuc)

    for i in range(n_endolysins):
        protein, nuc = _make_endolysin(rng)
        _add(f"lysin_{i+1:03d}", "endolysin peptidoglycan hydrolase", protein, nuc)

    for i in range(n_spanins):
        protein, nuc = _make_ispanin(rng)
        _add(f"rz_{i+1:03d}", "i-spanin Rz-like", protein, nuc)
        protein, nuc = _make_ospanin(rng)
        _add(f"rz1_{i+1:03d}", "o-spanin Rz1 lipoprotein", protein, nuc)

    # Scatter structural noise proteins at random positions
    for i in range(n_noise):
        protein, nuc = _make_structural_protein(rng)
        _add(f"orf_{i+1:03d}", "hypothetical protein", protein, nuc)

    # Shuffle noise proteins but keep lysis cluster together
    lysis  = [g for g in genes if g[1] != "hypothetical protein"]
    noise  = [g for g in genes if g[1] == "hypothetical protein"]
    rng.shuffle(noise)

    # Re-assign sequential positions after shuffling
    all_genes = []
    pos = 100
    for g in lysis + noise:
        start = pos
        protein_len = len(g[2])
        nuc_len     = len(g[3])
        end = pos + nuc_len - 4
        all_genes.append((g[1].replace(" ", "_")[:12] + f"_{len(all_genes)+1:03d}",
                           g[1], g[2], g[3], start, end, "+"))
        pos = end + rng.randint(10, 50)

    all_genes.sort(key=lambda g: g[4])
    return all_genes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic phage genome FASTAs for pipeline testing"
    )
    parser.add_argument("--output",       default="data/input/phage_genomes",
                        help="Output directory for FASTA files")
    parser.add_argument("--pharokka-out", default=None,
                        help="If set, also write mock Pharokka output here (for skipping M01)")
    parser.add_argument("--n-genomes",    type=int, default=3)
    parser.add_argument("--n-endolysins", type=int, default=2,
                        help="Endolysins per genome (default 2)")
    parser.add_argument("--n-holins",     type=int, default=2)
    parser.add_argument("--n-spanins",    type=int, default=1,
                        help="Spanin pairs (i+o) per genome")
    parser.add_argument("--n-noise",      type=int, default=12,
                        help="Noise structural proteins per genome")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_proteins  = []
    all_nucleotides = {}
    all_annotations = []

    for i in range(args.n_genomes):
        genome_id = f"test_phage_{i+1:03d}"
        print(f"Generating {genome_id}...")

        genes = generate_genome(
            genome_id    = genome_id,
            rng          = rng,
            n_endolysins = args.n_endolysins,
            n_holins     = args.n_holins,
            n_spanins    = args.n_spanins,
            n_noise      = args.n_noise,
        )

        # Write genome as single-sequence FASTA (concatenated ORFs, simplified)
        genome_seq = "ATGCATGC" * 500   # placeholder genome backbone
        fasta_path = out_dir / f"{genome_id}.fasta"
        with open(fasta_path, "w") as f:
            f.write(f">{genome_id} Synthetic test phage genome\n")
            f.write(genome_seq + "\n")

        print(f"  Written: {fasta_path} ({len(genes)} genes)")

        # Accumulate for mock Pharokka output
        for locus, product, protein, nuc, start, end, strand in genes:
            cid = f"{genome_id}__{locus}"
            all_proteins.append((cid, product, protein))
            all_nucleotides[cid] = nuc
            all_annotations.append({
                "candidate_id": cid, "genome_id": genome_id,
                "protein_id": locus, "function": product,
                "category": "lysis" if product != "hypothetical protein" else "unknown function",
                "start": start, "end": end, "strand": strand,
                "organism": genome_id,
            })

    # Write mock Pharokka outputs if requested (allows skipping M01 in testing)
    if args.pharokka_out:
        import csv, json
        pout = Path(args.pharokka_out) / "01_pharokka"
        pout.mkdir(parents=True, exist_ok=True)

        # all_proteins.faa
        with open(pout / "all_proteins.faa", "w") as f:
            for cid, product, protein in all_proteins:
                f.write(f">{cid} {product}\n{protein}\n")

        # all_nucleotides.ffn
        with open(pout / "all_nucleotides.ffn", "w") as f:
            for cid, nuc in all_nucleotides.items():
                f.write(f">{cid}\n{nuc}\n")

        # nucleotide_lookup.json
        (pout / "nucleotide_lookup.json").write_text(json.dumps(all_nucleotides, indent=2))

        # annotation_table.tsv
        if all_annotations:
            with open(pout / "annotation_table.tsv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_annotations[0].keys(), delimiter="\t")
                writer.writeheader()
                writer.writerows(all_annotations)

        n_endolysins = sum(1 for _, p, _ in all_proteins if "endolysin" in p.lower() or "lysin" in p.lower())
        n_holins     = sum(1 for _, p, _ in all_proteins if "holin" in p.lower())
        n_spanins    = sum(1 for _, p, _ in all_proteins if "spanin" in p.lower() or "rz" in p.lower())

        print(f"\nMock Pharokka output written to: {pout}")
        print(f"  Total proteins: {len(all_proteins)}")
        print(f"  Endolysins:     {n_endolysins}")
        print(f"  Holins:         {n_holins}")
        print(f"  Spanins:        {n_spanins}")
        print(f"  Noise:          {len(all_proteins) - n_endolysins - n_holins - n_spanins}")

    print(f"\nDone. {args.n_genomes} genome FASTAs written to {out_dir}")
    print("Run: python run_pipeline.py --config config/config.yaml")


if __name__ == "__main__":
    main()
