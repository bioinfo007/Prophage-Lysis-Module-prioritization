"""
m09_report.py
=============
Module 09: Generate all output files and per-candidate markdown reports.

Outputs:
  - priority_list.csv    : selected module components ranked by endolysin diversity rank
  - reserve_list.csv     : backup candidates
  - eliminated_log.csv   : full audit trail
  - pipeline_summary.json
  - candidate_reports/   : per-endolysin markdown summaries
  - umap_plot.png        : per-track 2D embedding visualization
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from pipeline.utils.data_model import (
    _BaseRecord, EndolysínRecord, HolinRecord, SpanínRecord,
    load_candidates,
)

log = logging.getLogger("m09_report")


def run(cfg: dict) -> None:
    paths   = cfg["paths"]
    rpt_cfg = cfg["reporting"]

    in_dir  = Path(paths["intermediate_dir"]) / "03_gate1"
    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_reports").mkdir(exist_ok=True)

    cand_path  = in_dir / "candidates_passing.json"
    candidates = load_candidates(str(cand_path))

    priority  = sorted(
        [c for c in candidates if c.final_status == "priority"],
        key=lambda c: (c.final_rank or 999, c.track != "endolysin"),
    )
    reserve   = [c for c in candidates if c.final_status == "reserve"]
    eliminated= [c for c in candidates if c.final_status == "eliminated"]

    log.info(
        f"Writing reports: {len(priority)} priority | "
        f"{len(reserve)} reserve | {len(eliminated)} eliminated"
    )

    _write_priority_list(priority,  out_dir / "priority_list.csv",  rpt_cfg)
    _write_reserve_list(reserve,    out_dir / "reserve_list.csv",   rpt_cfg)
    _write_eliminated_log(eliminated, out_dir / "eliminated_log.csv")
    _write_pipeline_summary(candidates, priority, reserve, eliminated,
                             out_dir / "pipeline_summary.json", cfg)

    if rpt_cfg.get("generate_candidate_reports", True):
        endolysin_priority = [c for c in priority if c.track == "endolysin"]
        for c in endolysin_priority:
            _write_candidate_report(c, out_dir / "candidate_reports")
        log.info(f"Written {len(endolysin_priority)} candidate reports")

    if rpt_cfg.get("generate_umap_plot", True):
        try:
            _generate_umap_plot(candidates, out_dir / "umap_plot.png")
        except ImportError:
            log.warning("matplotlib not available — skipping UMAP plot")
        except Exception as e:
            log.warning(f"UMAP plot failed: {e}")

    log.info(f"M09 complete — results in {out_dir}")


def _write_priority_list(
    candidates: List[_BaseRecord],
    path: Path,
    rpt_cfg: dict,
) -> None:
    columns = [
        "final_rank", "candidate_id", "track", "genome_id",
        "module_id", "module_complete", "pharokka_function",
        "pfam_domains", "length_aa", "mw_kda", "isoelectric_point",
        "gravy", "cai_score", "cluster_id", "diversity_rank",
        "novelty_flag", "closest_known", "sequence",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for c in candidates:
            row = []
            for col in columns:
                val = getattr(c, col, "")
                if isinstance(val, list):
                    val = "|".join(str(v) for v in val)
                row.append("" if val is None else val)
            writer.writerow(row)


def _write_reserve_list(
    candidates: List[_BaseRecord],
    path: Path,
    rpt_cfg: dict,
) -> None:
    columns = [
        "candidate_id", "track", "genome_id", "module_id",
        "pharokka_function", "pfam_domains", "mw_kda",
        "cluster_id", "similar_to", "max_similarity", "sequence",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for c in candidates:
            row = []
            for col in columns:
                val = getattr(c, col, "")
                if isinstance(val, list):
                    val = "|".join(str(v) for v in val)
                row.append("" if val is None else val)
            writer.writerow(row)


def _write_eliminated_log(candidates: List[_BaseRecord], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([
            "candidate_id", "track", "genome_id",
            "elimination_gate", "elimination_reason",
            "gate1_flags", "length_aa", "mw_kda",
        ])
        for c in candidates:
            writer.writerow([
                c.candidate_id, c.track, c.genome_id,
                c.elimination_gate or "",
                c.elimination_reason or "",
                "|".join(c.gate1_flags),
                c.length_aa or "",
                c.mw_kda or "",
            ])


def _write_pipeline_summary(
    candidates: List[_BaseRecord],
    priority:   List[_BaseRecord],
    reserve:    List[_BaseRecord],
    eliminated: List[_BaseRecord],
    path:       Path,
    cfg:        dict,
) -> None:
    def gate_count(gate: str) -> int:
        return sum(1 for c in eliminated if c.elimination_gate == gate)

    # Count by track
    def track_priority(t: str) -> int:
        return sum(1 for c in priority if c.track == t)

    summary = {
        "run_timestamp":        datetime.now().isoformat(),
        "pipeline_version":     "2.0.0",
        "total_candidates":     len(candidates),
        "priority_count":       len(priority),
        "reserve_count":        len(reserve),
        "eliminated_count":     len(eliminated),
        "eliminated_gate1":     gate_count("gate1"),
        "eliminated_gate3":     gate_count("gate3"),
        "priority_by_track": {
            "endolysin": track_priority("endolysin"),
            "holin":     track_priority("holin"),
            "spanin":    track_priority("spanin"),
        },
        "selection_strategy":   cfg["selection"].get("selection_strategy", "saturation"),
        "novel_endolysins":     sum(
            1 for c in priority
            if c.track == "endolysin" and c.novelty_flag == "novel"
        ),
        "complete_modules":     sum(1 for c in priority if c.module_complete),
        "priority_ids":         [c.candidate_id for c in priority],
    }
    path.write_text(json.dumps(summary, indent=2))


def _write_candidate_report(c: _BaseRecord, report_dir: Path) -> None:
    domains = ", ".join(c.pfam_domains) if c.pfam_domains else "none detected"
    flags   = ", ".join(c.gate1_flags)  if c.gate1_flags  else "none"

    is_endo   = isinstance(c, EndolysínRecord)
    is_sar    = is_endo and c.is_sar_endolysin
    is_novel  = c.novelty_flag == "novel"

    pg_table = ""
    if is_endo and c.pg_compatibility:
        pg_scores = c.get_pg_compatibility()
        pg_table  = "\n## PG Compatibility\n| Pathogen | Score |\n|---|---|\n"
        pg_table += "\n".join(
            f"| {pid} | {score} |" for pid, score in pg_scores.items()
        )

    content = f"""# {'⭐ NOVEL — ' if is_novel else ''}Candidate {c.candidate_id}
{'> **SAR endolysin** — signal-arrest-release mechanism' if is_sar else ''}

## Identity
| Field | Value |
|---|---|
| Candidate ID | `{c.candidate_id}` |
| Track | {c.track} |
| Genome | {c.genome_id} |
| Module | {c.module_id or 'unlinked'} ({'complete' if c.module_complete else 'partial'}) |
| Pharokka function | {c.pharokka_function} |

## Selection
| Field | Value |
|---|---|
| Final rank | {c.final_rank} |
| Diversity rank | {c.diversity_rank} |
| Min distance at selection | {c.min_dist_at_selection} |
| Novelty | {c.novelty_flag} |
| Closest known | {c.closest_known} |

## Domain architecture
| Domain | Present |
|---|---|
| CHAP (PF04851) | {'Yes' if is_endo and c.has_chap else 'No/N/A'} |
| Amidase (PF01520/PF13743) | {'Yes' if is_endo and c.has_amidase else 'No/N/A'} |
| Lysozyme (PF00959) | {'Yes' if is_endo and c.has_lysozyme else 'No/N/A'} |
| CBD | {'Yes' if is_endo and c.has_cbd else 'No/N/A'} |

All Pfam hits: {domains}
{pg_table}

## Physicochemical properties
| Property | Value |
|---|---|
| Length | {c.length_aa} aa |
| MW | {c.mw_kda} kDa |
| pI | {c.isoelectric_point} |
| GRAVY | {c.gravy} |
| Instability index | {c.instability_index} |
| CAI (E. coli) | {c.cai_score} |
| Gate 1 flags | {flags} |

## Sequence
```
{c.sequence}
```

---
*Generated {datetime.now().strftime('%Y-%m-%d')} — prophage_lysis v2.0.0*
"""
    (report_dir / f"{c.candidate_id}.md").write_text(content)


def _generate_umap_plot(candidates: List[_BaseRecord], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    TRACK_COLORS = {
        "endolysin": "#1D9E75",
        "holin":     "#377EB8",
        "spanin":    "#FF7F00",
    }
    STATUS_ALPHA  = {"priority": 1.0, "reserve": 0.4, "eliminated": 0.15}
    STATUS_SIZE   = {"priority": 80,  "reserve": 25,  "eliminated": 8}
    STATUS_ZORDER = {"priority": 4,   "reserve": 2,   "eliminated": 1}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    tracks = ["endolysin", "holin", "spanin"]

    for ax, track in zip(axes, tracks):
        track_cands = [c for c in candidates if c.track == track]
        if not track_cands:
            ax.set_title(f"{track.capitalize()}\n(no data)")
            continue

        for status in ["eliminated", "reserve", "priority"]:
            group = [c for c in track_cands
                     if c.final_status == status and c.umap_x is not None]
            if not group:
                continue
            ax.scatter(
                [c.umap_x for c in group],
                [c.umap_y for c in group],
                c      = TRACK_COLORS[track],
                s      = STATUS_SIZE[status],
                alpha  = STATUS_ALPHA[status],
                zorder = STATUS_ZORDER[status],
                label  = status.capitalize() if status == "priority" else None,
            )

        # Label priority by rank
        for c in track_cands:
            if c.final_status == "priority" and c.final_rank and c.umap_x is not None:
                ax.annotate(
                    str(c.final_rank),
                    (c.umap_x, c.umap_y),
                    fontsize=7, ha="center", va="center",
                    color="white", fontweight="bold", zorder=5,
                )

        n_pri = sum(1 for c in track_cands if c.final_status == "priority")
        n_res = sum(1 for c in track_cands if c.final_status == "reserve")
        n_eli = sum(1 for c in track_cands if c.final_status == "eliminated")
        ax.set_title(
            f"{track.capitalize()}\n"
            f"Priority:{n_pri} Reserve:{n_res} Eliminated:{n_eli}",
            fontsize=11,
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.suptitle(
        "Lysis Module Candidate Landscape\nESM-2 embeddings — UMAP 2D projection",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"UMAP plot saved: {output_path}")


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
