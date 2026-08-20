"""
scripts/benchmark.py
=====================
Measure per-module runtime and peak memory for profiling and paper methods table.

Usage:
  python scripts/benchmark.py --config config/config.yaml --modules 3 4 5 7 8
  python scripts/benchmark.py --config config/config.yaml --all
  python scripts/benchmark.py --config config/config.yaml --output benchmark_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


MODULE_NAMES = {
    1: "M01 Pharokka annotation",
    2: "M02 Lysis module identification",
    3: "M03 Gate 1 expressibility",
    4: "M04 ESM-2 embeddings",
    5: "M05 UMAP + HDBSCAN clustering",
    6: "M06 PG chemistry matching",
    7: "M07 Redundancy collapse",
    8: "M08 MaxMin selection",
    9: "M09 Report generation",
}


def benchmark_module(n: int, cfg: dict) -> Dict:
    """Run one module and measure wall time + peak memory."""
    from run_pipeline import _get_module
    from pipeline.utils.logging_config import setup_logging
    setup_logging(cfg["paths"].get("log_dir", "logs"), level="WARNING")

    module = _get_module(n)

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        module.run(cfg)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "module":       n,
            "name":         MODULE_NAMES.get(n, f"M{n:02d}"),
            "status":       "ok",
            "wall_time_s":  round(elapsed, 2),
            "peak_ram_mb":  round(peak / 1024 / 1024, 1),
        }
    except Exception as e:
        tracemalloc.stop()
        return {
            "module":   n,
            "name":     MODULE_NAMES.get(n, f"M{n:02d}"),
            "status":   "error",
            "error":    str(e)[:200],
        }


def format_table(results: List[Dict]) -> str:
    rows = ["Module                          | Time (s) | Peak RAM (MB) | Status"]
    rows.append("-" * 72)
    for r in results:
        name   = r.get("name", r["module"])[:32]
        time_s = f"{r.get('wall_time_s', '--'):>8.1f}" if r.get("status") == "ok" else "       --"
        ram    = f"{r.get('peak_ram_mb', '--'):>13.0f}" if r.get("status") == "ok" else "           --"
        status = r.get("status", "?")
        rows.append(f"{name:<32} | {time_s} | {ram} | {status}")
    rows.append("-" * 72)
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        total = sum(r.get("wall_time_s", 0) for r in ok)
        max_ram = max(r.get("peak_ram_mb", 0) for r in ok)
        rows.append(f"{'TOTAL':<32} | {total:>8.1f} | {max_ram:>13.0f} | --")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark prophage_lysis pipeline modules"
    )
    parser.add_argument("--config",  "-c", required=True)
    parser.add_argument("--modules", "-m", nargs="+", type=int, default=None,
                        help="Module numbers to benchmark (default: all)")
    parser.add_argument("--all",     action="store_true",
                        help="Benchmark all modules 1–9")
    parser.add_argument("--output",  "-o", default=None,
                        help="Save results JSON to this file")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    modules = list(range(1, 10)) if args.all else (args.modules or list(range(3, 10)))

    print(f"Benchmarking modules: {modules}")
    print("=" * 72)

    results = []
    for n in modules:
        name = MODULE_NAMES.get(n, f"M{n:02d}")
        print(f"Running {name}...", flush=True)
        r = benchmark_module(n, cfg)
        results.append(r)
        if r["status"] == "ok":
            print(f"  Done: {r['wall_time_s']:.1f}s, {r['peak_ram_mb']:.0f} MB peak RAM")
        else:
            print(f"  Error: {r.get('error', 'unknown')}")

    print("\n" + format_table(results))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
