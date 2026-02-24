# scripts/05_compare_results.py
"""
Reads graph build + algorithm timing JSONs from out/ and prints a neat speedup table.

Expected files (from earlier scripts):
  Graph build:
    out/graph_build_parallel_w1.json
    out/graph_build_parallel_w{N}.json

  Algo runs:
    out/algo_results/single/summary.json
    out/algo_results/parallel/summary.json

Usage (PowerShell):
  python scripts/05_compare_results.py --out out --workers 4

You can pass multiple workers:
  python scripts/05_compare_results.py --out out --workers 2 4 6
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_sec(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x < 1:
        return f"{x:.3f}s"
    if x < 60:
        return f"{x:.2f}s"
    if x < 3600:
        m = int(x // 60)
        s = x - 60 * m
        return f"{m}m {s:.1f}s"
    h = int(x // 3600)
    m = int((x - 3600 * h) // 60)
    s = x - 3600 * h - 60 * m
    return f"{h}h {m}m {s:.0f}s"


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _fmt_speedup(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}×"


def _print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    print("\n" + title)
    print("-" * len(title))
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def line(vals: List[str]) -> str:
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def load_graph_timings(out_dir: str, workers_list: List[int]) -> Tuple[Optional[float], List[Tuple[int, Optional[float]]]]:
    base = _read_json(os.path.join(out_dir, "graph_build_parallel_w1.json"))
    base_t = base.get("seconds_total") if base else None

    items = []
    for w in workers_list:
        p = os.path.join(out_dir, f"graph_build_parallel_w{w}.json")
        j = _read_json(p)
        t = j.get("seconds_total") if j else None
        items.append((w, t))
    return base_t, items


def load_algo_timings(out_dir: str) -> Tuple[Optional[float], Dict[str, Optional[float]], Optional[float], Dict[str, Optional[float]]]:
    single = _read_json(os.path.join(out_dir, "algo_results", "single", "summary.json"))
    parallel = _read_json(os.path.join(out_dir, "algo_results", "parallel", "summary.json"))

    def extract(summary: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        if not summary:
            return None, {}
        total = summary.get("total_seconds")
        per = {}
        for r in summary.get("results", []):
            algo = r.get("algo")
            sec = r.get("seconds")
            if algo:
                per[algo] = sec
        return total, per

    s_total, s_per = extract(single)
    p_total, p_per = extract(parallel)
    return s_total, s_per, p_total, p_per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="Output directory (default: out)")
    ap.add_argument("--workers", nargs="+", type=int, default=[4], help="Worker counts to compare (default: 4)")
    args = ap.parse_args()

    out_dir = args.out
    workers_list = sorted(set(args.workers))

    # --- Graph build table ---
    base_t, items = load_graph_timings(out_dir, workers_list)
    graph_rows = []
    for w, t in items:
        sp = _safe_div(base_t, t)
        graph_rows.append([str(w), _fmt_sec(t), _fmt_speedup(sp)])

    # include baseline row if present
    if base_t is not None:
        graph_rows.insert(0, ["1 (baseline)", _fmt_sec(base_t), "1.00×"])
    _print_table(
        "Graph Build Speedup",
        ["Workers", "Time", "Speedup vs 1"],
        graph_rows
    )

    # --- Algo execution table ---
    s_total, s_per, p_total, p_per = load_algo_timings(out_dir)

    algos = sorted(set(list(s_per.keys()) + list(p_per.keys())))
    algo_rows = []

    for a in algos:
        st = s_per.get(a)
        pt = p_per.get(a)
        algo_rows.append([
            a,
            _fmt_sec(st),
            _fmt_sec(pt),
            _fmt_speedup(_safe_div(st, pt))
        ])

    # totals
    if s_total is not None or p_total is not None:
        algo_rows.append([
            "TOTAL",
            _fmt_sec(s_total),
            _fmt_sec(p_total),
            _fmt_speedup(_safe_div(s_total, p_total))
        ])

    _print_table(
        "Algorithm Execution Speedup",
        ["Algo", "Single (sequential)", "Parallel (processes)", "Speedup"],
        algo_rows
    )

    # --- Helpful missing file hints ---
    missing = []
    if base_t is None:
        missing.append(os.path.join(out_dir, "graph_build_parallel_w1.json"))
    for w, t in items:
        if t is None:
            missing.append(os.path.join(out_dir, f"graph_build_parallel_w{w}.json"))
    if s_total is None:
        missing.append(os.path.join(out_dir, "algo_results", "single", "summary.json"))
    if p_total is None:
        missing.append(os.path.join(out_dir, "algo_results", "parallel", "summary.json"))

    if missing:
        print("\nMissing files (run the corresponding scripts first):")
        for m in missing:
            print(" -", m)


if __name__ == "__main__":
    main()