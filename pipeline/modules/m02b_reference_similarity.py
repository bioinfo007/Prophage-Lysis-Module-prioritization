"""
m02b_reference_similarity.py
=============================
Stage 3 lysis module identification: ESM-2 reference similarity.

Runs after M02 (HMMER + keyword) to catch unannotated endolysins, holins,
and spanins that have no Pfam domain hit and no keyword annotation.

Strategy:
  1. Embed ALL proteins from the phage genome with ESM-2
  2. Compute cosine similarity to curated reference sets
     (built by scripts/build_reference_embeddings.py)
  3. Proteins above threshold → add to candidate pool
  4. Already-identified proteins (from M02) skip this step

This catches the class of proteins that are:
  - Annotated as "hypothetical protein" by Pharokka
  - Have no significant Pfam hit
  - But are functionally similar to known endolysins in embedding space

Input:
  data/intermediate/01_pharokka/all_proteins.faa
  data/intermediate/02_lysis_modules/candidates.json   (existing M02 candidates)
  references/{endolysin,holin,spanin}_reference.npy    (built separately)

Output:
  data/intermediate/02_lysis_modules/candidates.json   (updated with new finds)
  data/intermediate/02_lysis_modules/ref_similarity_scores.tsv
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from Bio import SeqIO
from tqdm import tqdm

from pipeline.utils.data_model import (
    EndolysínRecord, HolinRecord, SpanínRecord,
    _BaseRecord, load_candidates, save_candidates,
)
from pipeline.utils.reference_db import ReferenceEmbeddingDB

log = logging.getLogger("m02b_reference_similarity")

ESM2_DIM = 1280
ESM2_MAX_LEN = 1022


def run(cfg: dict) -> int:
    """
    Run ESM-2 reference similarity search on all phage proteins.
    Returns number of new candidates added.
    """
    paths   = cfg["paths"]
    ref_dir = cfg.get("reference_db", "references")

    in_dir  = Path(paths["intermediate_dir"]) / "01_pharokka"
    m02_dir = Path(paths["intermediate_dir"]) / "02_lysis_modules"

    faa_path   = in_dir / "all_proteins.faa"
    nuc_lookup = json.loads((in_dir / "nucleotide_lookup.json").read_text()) \
        if (in_dir / "nucleotide_lookup.json").exists() else {}

    # Load reference DB
    db = ReferenceEmbeddingDB(ref_dir, thresholds=cfg.get("reference_thresholds"))

    if not db.any_available():
        log.warning(
            "No reference embedding sets found. "
            f"Run: python scripts/build_reference_embeddings.py --output {ref_dir}"
        )
        return 0

    log.info(db.summary())

    # Load existing M02 candidates (to avoid re-adding known proteins)
    existing_candidates = load_candidates(str(m02_dir / "candidates.json"))
    already_found: Set[str] = {c.candidate_id for c in existing_candidates}
    log.info(f"Existing M02 candidates: {len(already_found)}")

    # Load all proteins
    all_proteins: Dict[str, str] = {}
    for rec in SeqIO.parse(faa_path, "fasta"):
        pid = rec.id.split()[0]
        all_proteins[pid] = str(rec.seq).replace("*", "")

    log.info(f"Total phage proteins: {len(all_proteins)}")

    # Proteins not yet identified — these are the candidates for Stage 3
    unidentified = {
        pid: seq for pid, seq in all_proteins.items()
        if pid not in already_found and len(seq) >= 40
    }
    log.info(f"Unidentified proteins to screen: {len(unidentified)}")

    if not unidentified:
        log.info("All proteins already identified — skipping Stage 3")
        return 0

    # Embed all unidentified proteins
    embeddings, valid_ids = _embed_all(unidentified, cfg)
    if embeddings is None or len(valid_ids) == 0:
        log.error("Embedding failed — cannot run reference similarity")
        return 0

    log.info(f"Embedded {len(valid_ids)} proteins")

    # Score against each reference class
    new_candidates: List[_BaseRecord] = []
    score_rows: List[Dict] = []

    for class_name in ["endolysin", "holin", "spanin"]:
        if not db.is_available(class_name):
            log.info(f"  {class_name}: reference not available — skipping")
            continue

        scores = db.batch_score(class_name, embeddings, valid_ids, top_k=3)

        for score, pid in zip(scores, valid_ids):
            score_rows.append({
                "protein_id":     pid,
                "class":          class_name,
                "max_similarity": round(score["max_similarity"], 4),
                "is_candidate":   score["is_candidate"],
                "nearest_ref":    score["nearest"][0]["name"] if score["nearest"] else "",
                "nearest_sim":    round(score["nearest"][0]["similarity"], 4)
                                  if score["nearest"] else 0.0,
            })

            if not score["is_candidate"]:
                continue
            if pid in already_found:
                continue

            # Mark as found — don't add same protein twice even if it scores
            # above threshold for both endolysin and holin
            already_found.add(pid)

            seq    = unidentified[pid]
            parts  = pid.split("__", 1)
            genome_id = parts[0] if len(parts) == 2 else "unknown"
            locus     = parts[1] if len(parts) == 2 else pid

            nearest_str = "; ".join(
                f"{h['name']} ({h['similarity']:.3f})"
                for h in score["nearest"]
            )

            log.info(
                f"  [FOUND] {pid} → {class_name} "
                f"(similarity={score['max_similarity']:.3f}, "
                f"nearest: {score['nearest'][0]['name'] if score['nearest'] else 'unknown'})"
            )

            if class_name == "endolysin":
                c = EndolysínRecord(
                    candidate_id      = pid,
                    genome_id         = genome_id,
                    protein_id        = locus,
                    sequence          = seq,
                    nucleotide_seq    = nuc_lookup.get(pid, ""),
                    track             = "endolysin",
                    pharokka_function = "hypothetical protein",
                    inclusion_reason  = (
                        f"esm2_reference_similarity:{class_name}:"
                        f"{score['max_similarity']:.3f}:{nearest_str}"
                    ),
                    length_aa         = len(seq),
                )
            elif class_name == "holin":
                c = HolinRecord(
                    candidate_id      = pid,
                    genome_id         = genome_id,
                    protein_id        = locus,
                    sequence          = seq,
                    nucleotide_seq    = nuc_lookup.get(pid, ""),
                    track             = "holin",
                    pharokka_function = "hypothetical protein",
                    inclusion_reason  = (
                        f"esm2_reference_similarity:{class_name}:"
                        f"{score['max_similarity']:.3f}:{nearest_str}"
                    ),
                    length_aa         = len(seq),
                )
            else:  # spanin
                c = SpanínRecord(
                    candidate_id      = pid,
                    genome_id         = genome_id,
                    protein_id        = locus,
                    sequence          = seq,
                    nucleotide_seq    = nuc_lookup.get(pid, ""),
                    track             = "spanin",
                    pharokka_function = "hypothetical protein",
                    inclusion_reason  = (
                        f"esm2_reference_similarity:{class_name}:"
                        f"{score['max_similarity']:.3f}:{nearest_str}"
                    ),
                    length_aa         = len(seq),
                )

            new_candidates.append(c)

    # Write similarity scores TSV for all screened proteins
    _write_scores(score_rows, m02_dir / "ref_similarity_scores.tsv")

    # Merge with existing candidates and save
    if new_candidates:
        all_candidates = existing_candidates + new_candidates
        save_candidates(all_candidates, str(m02_dir / "candidates.json"))

        # Update candidates.faa
        with open(m02_dir / "candidates.faa", "a") as f:
            for c in new_candidates:
                f.write(f">{c.candidate_id} track={c.track} source=esm2_reference\n")
                f.write(f"{c.sequence}\n")

        log.info(
            f"Stage 3 complete — {len(new_candidates)} new candidates added "
            f"({sum(1 for c in new_candidates if c.track=='endolysin')} endolysins, "
            f"{sum(1 for c in new_candidates if c.track=='holin')} holins, "
            f"{sum(1 for c in new_candidates if c.track=='spanin')} spanins)"
        )
    else:
        log.info(
            "Stage 3 complete — no additional candidates above threshold. "
            "These phages may use non-canonical lysis mechanisms."
        )

    return len(new_candidates)


def _embed_all(
    proteins:   Dict[str, str],
    cfg:        dict,
) -> tuple:
    """Embed all proteins using ESM-2. Returns (matrix, id_list) or (None, [])."""
    backend = cfg.get("api", {}).get("esm2_backend", "local")

    if backend == "local":
        return _embed_local(proteins, cfg)
    else:
        return _embed_atlas(proteins, cfg)


def _embed_local(
    proteins:  Dict[str, str],
    cfg:       dict,
) -> tuple:
    try:
        import torch
        import esm as esm_lib
    except ImportError:
        log.error("Local ESM-2 not installed. Run: pip install torch fair-esm")
        return None, []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = cfg.get("api", {}).get("local_batch_size", 8)

    # Load model (cached after first call)
    log.info("Loading ESM-2 for Stage 3 reference similarity...")
    model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    sorted_items = sorted(proteins.items(), key=lambda x: len(x[1]))
    batches = [sorted_items[i:i+batch_size]
               for i in range(0, len(sorted_items), batch_size)]

    all_embeddings = []
    all_ids        = []

    for batch in tqdm(batches, desc="Stage 3 embedding", unit="batch"):
        data = [(pid, seq[:ESM2_MAX_LEN]) for pid, seq in batch]
        try:
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            with torch.no_grad():
                results = model(tokens, repr_layers=[33])
            reps = results["representations"][33]
            for i, (pid, seq) in enumerate(batch):
                seq_len = min(len(seq), ESM2_MAX_LEN)
                emb = reps[i, 1:seq_len+1].mean(0).cpu().numpy().astype(np.float32)
                if np.linalg.norm(emb) > 1e-6:
                    all_embeddings.append(emb)
                    all_ids.append(pid)
        except Exception as e:
            log.warning(f"Batch failed: {e}")

    if not all_embeddings:
        return None, []

    return np.vstack(all_embeddings).astype(np.float32), all_ids


def _embed_atlas(
    proteins: Dict[str, str],
    cfg:      dict,
) -> tuple:
    import requests
    import time

    atlas_url = cfg.get("api", {}).get(
        "esm_atlas_embed_url", "https://api.esmatlas.com/embedSequence/v1/"
    )
    retries = cfg.get("api", {}).get("retry_backoff", [5, 15, 45])
    timeout = cfg.get("api", {}).get("request_timeout", 120)

    embeddings = []
    valid_ids  = []

    for pid, seq in tqdm(proteins.items(), desc="Stage 3 Atlas", unit="protein"):
        seq_trunc = seq[:ESM2_MAX_LEN]
        last_exc  = None
        for wait in [0] + retries:
            if wait:
                time.sleep(wait)
            try:
                r = requests.post(
                    atlas_url,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    data=seq_trunc,
                    timeout=timeout,
                )
                r.raise_for_status()
                emb = np.array(r.json(), dtype=np.float32)
                if emb.shape[0] == ESM2_DIM and np.linalg.norm(emb) > 1e-6:
                    embeddings.append(emb)
                    valid_ids.append(pid)
                break
            except Exception as e:
                last_exc = e
        else:
            log.warning(f"  Atlas failed for {pid}: {last_exc}")

    if not embeddings:
        return None, []
    return np.vstack(embeddings).astype(np.float32), valid_ids


def _write_scores(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


# ── Standalone entry point ────────────────────────────────────────────────────
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
