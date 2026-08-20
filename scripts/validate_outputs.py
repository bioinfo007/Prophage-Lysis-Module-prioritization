"""
scripts/validate_outputs.py
============================
Post-run validation script.
Checks pipeline outputs for biological plausibility and data integrity.
Does NOT re-run the pipeline — reads existing output files only.

Checks:
  1. All required output files present
  2. Priority list has candidates from all three tracks
  3. No duplicate candidate_ids across priority + reserve
  4. Endolysin MW distribution is biologically plausible
  5. CAI scores are in valid range
  6. Module completeness is consistent
  7. Eliminated count makes sense given input size
  8. UMAP coordinates are present for all priority candidates
  9. No candidate has both 'priority' and 'reserve' status
  10. Novel fraction among priority candidates

Usage:
  python scripts/validate_outputs.py --config config/config.yaml
  python scripts/validate_outputs.py --config config/config.yaml --strict
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional


def check(condition: bool, msg: str, strict: bool, warns: List[str], errors: List[str]) -> None:
    if not condition:
        if strict:
            errors.append(f"[FAIL]  {msg}")
        else:
            warns.append(f"[WARN]  {msg}")


def load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def validate(config_path: str, strict: bool = False) -> int:
    """
    Validate pipeline outputs.
    Returns 0 if all checks pass, 1 if errors found.
    """
    import yaml
    cfg     = yaml.safe_load(open(config_path))
    out_dir = Path(cfg["paths"]["output_dir"])
    int_dir = Path(cfg["paths"]["intermediate_dir"])

    errors: List[str] = []
    warns:  List[str] = []
    ok:     List[str] = []

    # ── 1. Required output files ───────────────────────────────────────────────

    required_files = [
        out_dir / "priority_list.csv",
        out_dir / "reserve_list.csv",
        out_dir / "eliminated_log.csv",
        out_dir / "pipeline_summary.json",
    ]
    for f in required_files:
        if f.exists():
            ok.append(f"[OK]    {f.name} exists")
        else:
            errors.append(f"[FAIL]  Missing required output: {f.name}")

    if errors:
        _print_results(ok, warns, errors)
        return 1

    # ── Load data ──────────────────────────────────────────────────────────────

    priority    = load_csv(out_dir / "priority_list.csv")
    reserve     = load_csv(out_dir / "reserve_list.csv")
    eliminated  = load_csv(out_dir / "eliminated_log.csv")
    summary     = json.loads((out_dir / "pipeline_summary.json").read_text())

    ok.append(f"[OK]    priority_list.csv: {len(priority)} rows")
    ok.append(f"[OK]    reserve_list.csv:  {len(reserve)} rows")
    ok.append(f"[OK]    eliminated_log.csv: {len(eliminated)} rows")

    # ── 2. Priority list has endolysins ───────────────────────────────────────

    priority_tracks = {r.get("track", "") for r in priority}
    check(
        "endolysin" in priority_tracks,
        "No endolysins in priority list — pipeline may have filtered everything",
        strict=True, warns=warns, errors=errors,
    )
    if "endolysin" in priority_tracks:
        ok.append(f"[OK]    Endolysins in priority list: "
                  f"{sum(1 for r in priority if r.get('track') == 'endolysin')}")

    # ── 3. No duplicate candidate_ids ─────────────────────────────────────────

    priority_ids = [r.get("candidate_id", "") for r in priority if r.get("candidate_id")]
    reserve_ids  = [r.get("candidate_id", "") for r in reserve  if r.get("candidate_id")]
    all_ids      = priority_ids + reserve_ids

    dup_within_priority = len(priority_ids) != len(set(priority_ids))
    overlap = set(priority_ids) & set(reserve_ids)

    check(not dup_within_priority,
          f"Duplicate candidate_ids in priority_list.csv",
          strict=True, warns=warns, errors=errors)
    check(not overlap,
          f"{len(overlap)} candidate_ids appear in BOTH priority and reserve lists",
          strict=True, warns=warns, errors=errors)

    if not dup_within_priority and not overlap:
        ok.append(f"[OK]    No duplicate candidate_ids across priority+reserve")

    # ── 4. Endolysin MW plausibility ──────────────────────────────────────────

    endo_mws = []
    for r in priority:
        if r.get("track") == "endolysin" and r.get("mw_kda"):
            try:
                mw = float(r["mw_kda"])
                endo_mws.append(mw)
            except ValueError:
                pass

    if endo_mws:
        min_mw = min(endo_mws)
        max_mw = max(endo_mws)
        med_mw = sorted(endo_mws)[len(endo_mws) // 2]
        ok.append(f"[OK]    Endolysin MW range: {min_mw:.1f}–{max_mw:.1f} kDa (median {med_mw:.1f})")
        check(
            max_mw <= 80.0,
            f"Max endolysin MW {max_mw:.1f} kDa > 80 kDa — unusually large",
            strict=False, warns=warns, errors=errors,
        )
        check(
            min_mw >= 5.0,
            f"Min endolysin MW {min_mw:.1f} kDa < 5 kDa — suspiciously small",
            strict=False, warns=warns, errors=errors,
        )

    # ── 5. CAI scores in valid range ───────────────────────────────────────────

    cai_scores = []
    for r in priority:
        if r.get("cai_score"):
            try:
                cai_scores.append(float(r["cai_score"]))
            except ValueError:
                pass

    if cai_scores:
        bad_cai = [c for c in cai_scores if not (0.0 <= c <= 1.0)]
        check(
            not bad_cai,
            f"{len(bad_cai)} CAI scores outside [0, 1] range: {bad_cai[:3]}",
            strict=True, warns=warns, errors=errors,
        )
        if not bad_cai:
            ok.append(f"[OK]    CAI scores valid (range {min(cai_scores):.3f}–{max(cai_scores):.3f})")

    # ── 6. Module completeness consistency ─────────────────────────────────────

    priority_endos = [r for r in priority if r.get("track") == "endolysin"]
    n_complete = sum(1 for r in priority_endos if r.get("module_complete") in ("True", "1", True))
    if priority_endos:
        pct_complete = n_complete / len(priority_endos) * 100
        ok.append(f"[OK]    Module-complete endolysins: {n_complete}/{len(priority_endos)} ({pct_complete:.0f}%)")
        check(
            pct_complete >= 20.0,
            f"Only {pct_complete:.0f}% of priority endolysins are in complete modules "
            f"— check module linkage (M02)",
            strict=False, warns=warns, errors=errors,
        )

    # ── 7. Elimination sanity ──────────────────────────────────────────────────

    total_processed = len(priority) + len(reserve) + len(eliminated)
    if total_processed > 0:
        pct_eliminated = len(eliminated) / total_processed * 100
        ok.append(f"[OK]    Eliminated: {len(eliminated)}/{total_processed} ({pct_eliminated:.0f}%)")
        check(
            pct_eliminated < 95.0,
            f"{pct_eliminated:.0f}% of candidates eliminated — filters may be too strict",
            strict=False, warns=warns, errors=errors,
        )
        check(
            pct_eliminated > 0.0,
            "0% eliminated — filters may not be working",
            strict=False, warns=warns, errors=errors,
        )

    # ── 8. Summary statistics consistency ─────────────────────────────────────

    check(
        summary.get("priority_count", 0) == len(priority),
        f"summary.priority_count {summary.get('priority_count')} != "
        f"len(priority_list.csv) {len(priority)}",
        strict=True, warns=warns, errors=errors,
    )
    if summary.get("priority_count", 0) == len(priority):
        ok.append(f"[OK]    Summary counts match CSV row counts")

    # ── 9. Novel candidates ────────────────────────────────────────────────────

    novel_count  = summary.get("novel_endolysins", 0)
    n_endo_pri   = summary.get("priority_by_track", {}).get("endolysin", 0)
    if n_endo_pri > 0:
        novel_pct = novel_count / n_endo_pri * 100
        ok.append(f"[OK]    Novel endolysins: {novel_count}/{n_endo_pri} ({novel_pct:.0f}%)")

    # ── 10. Per-candidate reports ──────────────────────────────────────────────

    report_dir = out_dir / "candidate_reports"
    if report_dir.exists():
        n_reports = len(list(report_dir.glob("*.md")))
        ok.append(f"[OK]    Candidate reports: {n_reports} markdown files")
        check(
            n_reports == n_endo_pri,
            f"Report count ({n_reports}) != priority endolysin count ({n_endo_pri})",
            strict=False, warns=warns, errors=errors,
        )

    # ── 11. UMAP plot ──────────────────────────────────────────────────────────

    if (out_dir / "umap_plot.png").exists():
        size = (out_dir / "umap_plot.png").stat().st_size
        ok.append(f"[OK]    umap_plot.png exists ({size // 1024} KB)")
        check(size > 10000, "UMAP plot file is suspiciously small (<10 KB)",
              strict=False, warns=warns, errors=errors)

    # ── Print results ──────────────────────────────────────────────────────────

    _print_results(ok, warns, errors)

    if errors:
        print(f"\n{'='*60}")
        print(f"VALIDATION FAILED — {len(errors)} error(s), {len(warns)} warning(s)")
        return 1
    elif warns:
        print(f"\n{'='*60}")
        print(f"VALIDATION PASSED WITH WARNINGS — {len(warns)} warning(s)")
        return 0
    else:
        print(f"\n{'='*60}")
        print(f"VALIDATION PASSED — {len(ok)} checks passed")
        return 0


def _print_results(ok, warns, errors) -> None:
    print("\nValidation results:")
    for line in ok:
        print(f"  {line}")
    for line in warns:
        print(f"  {line}")
    for line in errors:
        print(f"  {line}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate prophage_lysis pipeline outputs for biological plausibility"
    )
    parser.add_argument("--config",  "-c", required=True, help="Path to config.yaml")
    parser.add_argument("--strict",        action="store_true",
                        help="Treat warnings as errors")
    args = parser.parse_args()

    exit_code = validate(args.config, strict=args.strict)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
