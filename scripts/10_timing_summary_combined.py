import os
import sys
import json
import argparse
from typing import Any, Dict, Optional, List

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

def extract_cpu_algos(summary_json):
    out = {}
    if not summary_json:
        return out
    for r in summary_json.get("results", []):
        algo = r.get("algo")
        compute = r.get("seconds_compute")
        if compute is None:
            compute = r.get("seconds")
        load = r.get("seconds_load_graph")
        total = r.get("seconds_total")
        if total is None:
            total = (compute or 0) + (load or 0)
        out[algo] = {
            "compute": compute,
            "load": load,
            "algo_total": total
        }
    return out

def extract_gpu_algos(gpu_json):
    out = {}
    if not gpu_json:
        return out, None
    transfer = gpu_json.get("seconds_transfer_to_gpu")
    runs = gpu_json.get("runs", [])
    for r in runs:
        algo = r.get("algo")
        compute = r.get("seconds_compute_gpu")
        if compute is None:
            compute = r.get("seconds")
        out[algo] = {
            "compute": compute,
            "transfer": transfer
        }
    return out, transfer

def print_table(headers: List[str], rows: List[List[str]]):
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
    ap.add_argument("--build_w1", required=True)
    ap.add_argument("--build_wN", required=True)
    ap.add_argument("--cpu_single", required=True)
    ap.add_argument("--cpu_parallel", required=True)
    ap.add_argument("--gpu", required=True)
    args = ap.parse_args()

    build_w1 = read_json(args.build_w1)
    build_wN = read_json(args.build_wN)
    cpu_s = read_json(args.cpu_single)
    cpu_p = read_json(args.cpu_parallel)
    gpu = read_json(args.gpu)

    csr_w1 = build_w1.get("seconds_total")
    csr_wN = build_wN.get("seconds_total")

    cpu_single = extract_cpu_algos(cpu_s)
    cpu_parallel = extract_cpu_algos(cpu_p)
    gpu_algos, gpu_transfer = extract_gpu_algos(gpu)

    algos = sorted(set(cpu_single) | set(cpu_parallel) | set(gpu_algos))

    rows = []

    for a in algos:
        cs = cpu_single.get(a, {})
        cp = cpu_parallel.get(a, {})
        gg = gpu_algos.get(a, {})

        total_cpu1 = (csr_w1 or 0) + (cs.get("algo_total") or 0)
        total_cpuN = (csr_wN or 0) + (cp.get("algo_total") or 0)
        total_gpu  = (csr_w1 or 0) + (gg.get("compute") or 0) + (gpu_transfer or 0)

        rows.append([
            a,
            fmt(total_cpu1),
            fmt(total_cpuN),
            fmt(total_gpu)
        ])

    print("\n=== FINAL TOTAL TIMINGS (CSR build + algorithm run) ===\n")
    print_table(
        ["Algorithm", "CPU single", "CPU parallel", "GPU"],
        rows
    )

if __name__ == "__main__":
    main()