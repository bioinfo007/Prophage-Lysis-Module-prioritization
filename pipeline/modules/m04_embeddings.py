"""
m04_embeddings.py
=================
Module 04: Generate ESM-2 protein language model embeddings.

Runs AFTER Gate 1 — only embeds candidates that passed.
This saves 30-50% compute compared to embedding before filtering.

Backends:
  "atlas"  — ESM Metagenomic Atlas REST API (free, no setup, needs internet)
  "local"  — Local ESM-2 model with batched inference (fast on CPU/GPU)

Optimizations:
  - Batched inference (grouped by sequence length to minimize padding waste)
  - Per-candidate .npy cache for safe resume
  - Manifest JSON tracks completed IDs (O(1) resume check, not filesystem scan)
  - Zero-vector candidates are excluded from matrix — not silently injected
  - Auto-detects GPU via PyTorch, falls back to CPU transparently

Input:  data/intermediate/03_gate1/candidates_passing.json
Output: data/intermediate/04_embeddings/{candidate_id}.npy
        data/intermediate/04_embeddings/embedding_matrix.npy
        data/intermediate/04_embeddings/embedding_index.json
        data/intermediate/04_embeddings/manifest.json           ← resume tracker
        data/intermediate/03_gate1/candidates_passing.json (updated)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from pipeline.utils.data_model import (
    _BaseRecord, load_candidates, save_candidates,
)

log = logging.getLogger("m04_embeddings")

ESM2_DIM = 1280
ESM2_MAX_LEN = 1022   # hard token limit for ESM-2


def run(cfg: dict) -> None:
    paths   = cfg["paths"]
    api_cfg = cfg["api"]

    in_dir  = Path(paths["intermediate_dir"]) / "03_gate1"
    out_dir = Path(paths["intermediate_dir"]) / "04_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path  = in_dir / "candidates_passing.json"
    candidates = load_candidates(str(cand_path))

    backend = api_cfg.get("esm2_backend", "atlas")

    # Validate backend choice
    if backend == "local":
        try:
            import torch
            import esm  # noqa: F401
        except ImportError as e:
            log.warning(
                f"Backend 'local' requested but dependency missing: {e}\n"
                f"Falling back to Atlas REST API.\n"
                f"To fix: pip install torch fair-esm\n"
                f"Or set api.esm2_backend: atlas in config.yaml"
            )
            backend = "atlas"

    log.info(
        f"ESM-2 embeddings: {len(candidates)} candidates | backend: {backend}"
    )

    # Load manifest (completed candidate IDs)
    manifest_path = out_dir / "manifest.json"
    manifest: Dict[str, str] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    already_done = set(manifest.keys())
    log.info(f"  {len(already_done)} already cached from previous run")

    # ── Local backend: load model once ────────────────────────────────────────
    local_ctx: Optional[dict] = None
    if backend == "local":
        local_ctx = _load_local_esm2()

    # ── Embed all candidates ──────────────────────────────────────────────────
    failed_ids: List[str] = []

    if backend == "local" and local_ctx:
        # Batched local embedding
        to_embed = [c for c in candidates if c.candidate_id not in already_done]
        log.info(f"  Batching {len(to_embed)} sequences for local ESM-2")
        _embed_local_batched(
            to_embed, local_ctx, out_dir, manifest, manifest_path, failed_ids,
            batch_size=api_cfg.get("local_batch_size", 16),
        )
    else:
        # Atlas API — one at a time with retry
        retries = api_cfg.get("retry_backoff", [5, 15, 45])
        timeout = api_cfg.get("request_timeout", 120)
        atlas_url = api_cfg.get(
            "esm_atlas_embed_url",
            "https://api.esmatlas.com/embedSequence/v1/",
        )
        for c in tqdm(candidates, desc="ESM-Atlas", unit="protein"):
            if c.candidate_id in already_done:
                continue
            try:
                emb = _atlas_embed(c.sequence, atlas_url, retries, timeout)
                _save_embedding(c, emb, out_dir, manifest, manifest_path)
            except Exception as e:
                log.error(f"  {c.candidate_id}: Atlas embed failed — {e}")
                failed_ids.append(c.candidate_id)

    # ── Build embedding matrix from cached files ──────────────────────────────
    log.info("Building embedding matrix from cached files...")
    valid_candidates: List[_BaseRecord] = []
    embeddings_list:  List[np.ndarray]  = []
    index:            List[str]         = []

    for c in candidates:
        emb_path = out_dir / f"{c.candidate_id}.npy"
        if not emb_path.exists():
            if c.candidate_id not in failed_ids:
                log.warning(f"  No embedding file for {c.candidate_id} — excluding")
            continue

        emb = np.load(emb_path)
        if emb.shape[0] != ESM2_DIM:
            log.warning(f"  {c.candidate_id}: wrong dim {emb.shape} — excluding")
            continue

        c.embedding_path = str(emb_path)
        c.embedding_dim  = ESM2_DIM
        embeddings_list.append(emb)
        index.append(c.candidate_id)
        valid_candidates.append(c)

    if not embeddings_list:
        raise RuntimeError(
            "No valid embeddings produced. "
            "Check API connectivity or local model setup."
        )

    matrix = np.vstack(embeddings_list).astype(np.float32)
    np.save(out_dir / "embedding_matrix.npy", matrix)
    (out_dir / "embedding_index.json").write_text(json.dumps(index, indent=2))

    # Update candidates file with embedding paths
    save_candidates(valid_candidates, str(cand_path))

    if failed_ids:
        log.warning(
            f"{len(failed_ids)} candidates failed embedding and were excluded: "
            f"{failed_ids[:5]}{'...' if len(failed_ids) > 5 else ''}"
        )

    log.info(
        f"M04 complete — matrix shape: {matrix.shape} | "
        f"{len(failed_ids)} excluded (embed failures)"
    )


# ── Local ESM-2 batched inference ─────────────────────────────────────────────

def _load_local_esm2() -> dict:
    # Check torch availability
    try:
        import torch
        _torch_ok = True
    except ImportError:
        _torch_ok = False

    # Check fair-esm availability
    try:
        import esm as esm_lib
        _esm_ok = True
    except ImportError:
        _esm_ok = False

    if not _torch_ok or not _esm_ok:
        missing = []
        if not _torch_ok: missing.append("torch")
        if not _esm_ok:   missing.append("fair-esm")
        raise ImportError(
            f"Local ESM-2 backend requires: pip install {' '.join(missing)}\n"
            f"\n"
            f"Easier fix — switch to the Atlas REST API (no install needed):\n"
            f"  Set in config.yaml:  api:\n"
            f"                         esm2_backend: atlas\n"
            f"The Atlas API is free, works on any machine, and requires no GPU."
        )

    log.info("Loading ESM-2 model (first run downloads ~1.5 GB)...")

    # GPU is optional — falls back to CPU silently
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.info("  Device: CPU (no CUDA GPU detected — inference will be slower)")
        log.info("  Tip: set api.esm2_backend: atlas for faster cloud inference")
    else:
        log.info(f"  Device: {device} (GPU detected)")

    model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    log.info("ESM-2 loaded successfully")
    return {
        "model":           model,
        "alphabet":        alphabet,
        "batch_converter": batch_converter,
        "device":          device,
    }


def _embed_local_batched(
    candidates:     List[_BaseRecord],
    ctx:            dict,
    out_dir:        Path,
    manifest:       Dict[str, str],
    manifest_path:  Path,
    failed_ids:     List[str],
    batch_size:     int = 16,
) -> None:
    """
    Batch ESM-2 inference grouped by sequence length bucket.
    Grouping by similar length minimizes padding waste and is 5-10x faster
    than one-at-a-time inference on CPU.
    """
    import torch

    device = ctx["device"]

    # Sort by length for batching efficiency
    sorted_cands = sorted(candidates, key=lambda c: len(c.sequence))

    # Split into batches
    batches: List[List[_BaseRecord]] = []
    for i in range(0, len(sorted_cands), batch_size):
        batches.append(sorted_cands[i:i + batch_size])

    for batch in tqdm(batches, desc="ESM-2 [local]", unit="batch"):
        data = []
        for c in batch:
            seq = c.sequence[:ESM2_MAX_LEN]   # hard truncate at token limit
            data.append((c.candidate_id, seq))

        try:
            _, _, tokens = ctx["batch_converter"](data)
            tokens = tokens.to(device)

            with torch.no_grad():
                results = ctx["model"](tokens, repr_layers=[33])

            reps = results["representations"][33]   # (B, L+2, 1280)

            for i, c in enumerate(batch):
                seq_len = min(len(c.sequence), ESM2_MAX_LEN)
                # Mean pool over sequence positions (exclude BOS/EOS tokens)
                emb = reps[i, 1:seq_len + 1].mean(0).cpu().numpy().astype(np.float32)
                _save_embedding(c, emb, out_dir, manifest, manifest_path)

        except Exception as e:
            log.error(f"Batch embedding failed: {e}")
            for c in batch:
                failed_ids.append(c.candidate_id)


# ── Atlas API embedding ───────────────────────────────────────────────────────

def _atlas_embed(
    sequence: str,
    api_url:  str,
    retries:  List[int],
    timeout:  int,
) -> np.ndarray:
    import requests

    seq = sequence[:ESM2_MAX_LEN]
    last_exc = None

    for attempt, wait in enumerate([0] + retries):
        if wait:
            log.debug(f"    Retry in {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
        try:
            r = requests.post(
                api_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=seq,
                timeout=timeout,
            )
            r.raise_for_status()
            result = r.json()
            if not isinstance(result, list):
                raise ValueError(f"Unexpected response: {str(result)[:100]}")
            return np.array(result, dtype=np.float32)
        except Exception as e:
            last_exc = e
            log.debug(f"    Atlas embed error: {e}")

    raise RuntimeError(f"Atlas embed failed after retries: {last_exc}")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _save_embedding(
    c:             _BaseRecord,
    emb:           np.ndarray,
    out_dir:       Path,
    manifest:      Dict[str, str],
    manifest_path: Path,
) -> None:
    emb_path = out_dir / f"{c.candidate_id}.npy"
    np.save(emb_path, emb)
    manifest[c.candidate_id] = str(emb_path)
    manifest_path.write_text(json.dumps(manifest))


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
