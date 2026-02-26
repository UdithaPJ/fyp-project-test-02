import os
import sys
import json
import argparse
from typing import Any, Dict, Optional, List, Tuple

def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fmt(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x < 1:
        return f"{x:.4f}s"
    if x < 60:
        return f"{x:.2f}s"
    m = int(x // 60)
    s = x - 60 * m
    return f"{m}m {s:.1f}s"

def get_graph_build_time(build_json: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    if not build_json:
        return None, {}
    total = build_json.get("seconds_total")
    parts = {
        "norms": build_json.get("seconds_norms"),
        "blocks": build_json.get("seconds_blocks"),
        "merge_save": build_json.get("seconds_merge_save") or build_json.get("seconds_merge") or build_json.get("seconds_merge_save"),
    }
    return total, parts

def extract_cpu_algos(summary_json: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Returns:
      algo -> {load, compute, total}
    Handles both old fields (seconds) and new fields (seconds_compute, seconds_load_graph).
    """
    out = {}
    if not summary_json:
        return out
    for r in summary_json.get("results", []):
        algo = r.get("algo")
        if not algo:
            continue
        load = r.get("seconds_load_graph")
        compute = r.get("seconds_compute")
        if compute is None:
            compute = r.get("seconds")  # older script
        total = r.get("seconds_total")
        if total is None:
            # if we have load+compute
            if load is not None and compute is not None:
                total = load + compute
            else:
                total = compute
        out[algo] = {"load": load, "compute": compute, "total": total}
    return out

def extract_gpu_algos(gpu_json: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[float], Optional[float]]:
    """
    Returns:
      algo -> {transfer_share, compute, total_including_transfer_share}
    plus transfer time if present (seconds_transfer_to_gpu) and cpu load time if present.
    """
    out = {}
    if not gpu_json:
        return out, None, None

    tx = gpu_json.get("seconds_transfer_to_gpu")
    load_cpu = gpu_json.get("seconds_load_csr_cpu")

    runs = gpu_json.get("runs", [])
    if not runs:
        return out, tx, load_cpu

    # If transfer exists, we can attribute it equally per algo for "total" reporting.
    per_algo_tx = (tx / len(runs)) if (tx is not None and len(runs) > 0) else None

    for r in runs:
        algo = r.get("algo")
        if not algo:
            continue
        compute = r.get("seconds_compute_gpu")
        if compute is None:
            compute = r.get("seconds")  # older gpu script
        total = compute
        if per_algo_tx is not None and total is not None:
            total = total + per_algo_tx
        out[algo] = {"transfer_share": per_algo_tx, "compute": compute, "total": total}

    return out, tx, load_cpu

def print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    print("\n" + title)
    print("-" * len(title))
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    def line(vals):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_w1", default="out/graph_build_parallel_w1.json", help="CSR build timing JSON for workers=1")
    ap.add_argument("--build_wN", default=None, help="CSR build timing JSON for workers=N (e.g., out/graph_build_parallel_w4.json)")
    ap.add_argument("--cpu_single", default="out/algo_results/single/summary.json", help="CPU single summary.json")
    ap.add_argument("--cpu_parallel", default="out/algo_results/parallel/summary.json", help="CPU parallel summary.json")
    ap.add_argument("--gpu", default="out/gpu_results/gpu_results.json", help="GPU gpu_results.json")
    ap.add_argument("--out_json", default=None, help="Optional: write combined summary JSON to this path")
    args = ap.parse_args()

    build1 = read_json(args.build_w1)
    buildN = read_json(args.build_wN) if args.build_wN else None
    cpu_s = read_json(args.cpu_single)
    cpu_p = read_json(args.cpu_parallel)
    gpu = read_json(args.gpu)

    csr1_total, csr1_parts = get_graph_build_time(build1)
    csrN_total, csrN_parts = get_graph_build_time(buildN) if buildN else (None, {})

    cpu_single_algos = extract_cpu_algos(cpu_s)
    cpu_parallel_algos = extract_cpu_algos(cpu_p)
    gpu_algos, gpu_tx, gpu_load_cpu = extract_gpu_algos(gpu)

    # determine algos to show (union)
    algos = sorted(set(cpu_single_algos.keys()) | set(cpu_parallel_algos.keys()) | set(gpu_algos.keys()))

    # CSR summary table
    csr_rows = [
        ["csr_build_w1", fmt(csr1_total), fmt(csr1_parts.get("norms")), fmt(csr1_parts.get("blocks")), fmt(csr1_parts.get("merge_save"))]
    ]
    if args.build_wN:
        csr_rows.append(["csr_build_wN", fmt(csrN_total), fmt(csrN_parts.get("norms")), fmt(csrN_parts.get("blocks")), fmt(csrN_parts.get("merge_save"))])

    print_table(
        "CSR Creation Timing",
        ["Mode", "Total", "Norms", "Blocks", "Merge+Save"],
        csr_rows
    )

    # Algo summary table (compute + load/transfer)
    rows = []
    for a in algos:
        cs = cpu_single_algos.get(a, {})
        cp = cpu_parallel_algos.get(a, {})
        gg = gpu_algos.get(a, {})

        rows.append([
            a,
            fmt(cs.get("compute")), fmt(cs.get("load")), fmt(cs.get("total")),
            fmt(cp.get("compute")), fmt(cp.get("load")), fmt(cp.get("total")),
            fmt(gg.get("compute")), fmt(gg.get("transfer_share")), fmt(gg.get("total")),
        ])

    print_table(
        "Algorithm Timing (per algorithm)",
        [
            "Algo",
            "CPU1_compute", "CPU1_load", "CPU1_total",
            "CPUp_compute", "CPUp_load", "CPUp_total",
            "GPU_compute", "GPU_tx_share", "GPU_total*"
        ],
        rows
    )

    if gpu is not None:
        print("\nGPU note:")
        print(f"  seconds_load_csr_cpu   : {fmt(gpu_load_cpu)}")
        print(f"  seconds_transfer_to_gpu: {fmt(gpu_tx)}")
        print("  GPU_total* = GPU_compute + (transfer time / #algos).")

    combined = {
        "inputs": {
            "build_w1": os.path.abspath(args.build_w1) if args.build_w1 else None,
            "build_wN": os.path.abspath(args.build_wN) if args.build_wN else None,
            "cpu_single": os.path.abspath(args.cpu_single) if args.cpu_single else None,
            "cpu_parallel": os.path.abspath(args.cpu_parallel) if args.cpu_parallel else None,
            "gpu": os.path.abspath(args.gpu) if args.gpu else None,
        },
        "csr_build": {
            "w1": {"total": csr1_total, **csr1_parts},
            "wN": {"total": csrN_total, **csrN_parts} if args.build_wN else None,
        },
        "algorithms": {
            a: {
                "cpu_single": cpu_single_algos.get(a),
                "cpu_parallel": cpu_parallel_algos.get(a),
                "gpu": gpu_algos.get(a),
            } for a in algos
        },
        "gpu_transfer_total": gpu_tx,
        "gpu_load_csr_cpu": gpu_load_cpu,
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print("\nWrote combined summary JSON:", args.out_json)

if __name__ == "__main__":
    main()