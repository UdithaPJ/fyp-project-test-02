import os, sys, json, time
import numpy as np
from scipy import sparse

# ---- CPU baselines (same logic as your CPU stage) ----
def cpu_pagerank(A, d=0.85, tol=1e-6, max_iter=200):
    n = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(out, dtype=np.float64)
    inv[out > 0] = 1.0 / out[out > 0]
    P = (sparse.diags(inv) @ A).tocsr()

    pr = np.full(n, 1.0/n, dtype=np.float64)
    teleport = (1.0 - d) / n
    for _ in range(max_iter):
        pr_new = teleport + d * (P.T @ pr)
        if np.abs(pr_new - pr).sum() < tol:
            pr = pr_new
            break
        pr = pr_new
    return pr

def cpu_rwr(A, seed=0, restart=0.5, tol=1e-6, max_iter=300):
    n = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(out, dtype=np.float64)
    inv[out > 0] = 1.0 / out[out > 0]
    P = (sparse.diags(inv) @ A).tocsr()

    p0 = np.zeros(n, dtype=np.float64)
    p0[int(seed)] = 1.0
    p = p0.copy()
    for _ in range(max_iter):
        p_new = (1.0 - restart) * (P.T @ p) + restart * p0
        if np.abs(p_new - p).sum() < tol:
            p = p_new
            break
        p = p_new
    return p

def cpu_hits(A, tol=1e-6, max_iter=300):
    n = A.shape[0]
    h = np.ones(n, dtype=np.float64)
    a = np.ones(n, dtype=np.float64)
    for _ in range(max_iter):
        a_new = A.T @ h
        h_new = A @ a_new
        a_new /= (np.linalg.norm(a_new) + 1e-12)
        h_new /= (np.linalg.norm(h_new) + 1e-12)
        if (np.linalg.norm(a_new - a) + np.linalg.norm(h_new - h)) < tol:
            a, h = a_new, h_new
            break
        a, h = a_new, h_new
    return a, h

def cpu_bfs(A, source=0):
    from collections import deque
    n = A.shape[0]
    dist = -np.ones(n, dtype=np.int32)
    q = deque([int(source)])
    dist[int(source)] = 0
    indptr, indices = A.indptr, A.indices
    while q:
        u = q.popleft()
        du = dist[u]
        for ei in range(indptr[u], indptr[u+1]):
            v = indices[ei]
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist

# ---- GPU results loader ----
def load_gpu_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python scripts/07_compare_cpu_gpu.py <graph_npz> <gpu_results_json> <out_json>")
        raise SystemExit(2)

    graph_path = sys.argv[1]
    gpu_json = sys.argv[2]
    out_json = sys.argv[3]

    A = sparse.load_npz(graph_path).tocsr()
    if A.dtype != np.float32:
        A = A.astype(np.float32)

    gpu = load_gpu_results(gpu_json)

    report = {
        "graph": os.path.abspath(graph_path),
        "nnz": int(A.nnz),
        "n": int(A.shape[0]),
        "cpu": [],
        "gpu": gpu.get("runs", []),
        "comparisons": []
    }

    # CPU timings
    t0 = time.time(); pr = cpu_pagerank(A); t_pr = time.time()-t0
    report["cpu"].append({"algo": "pagerank", "seconds": t_pr, "top20": np.argsort(pr)[-20:][::-1].tolist()})

    t0 = time.time(); r = cpu_rwr(A); t_r = time.time()-t0
    report["cpu"].append({"algo": "rwr", "seconds": t_r, "top20": np.argsort(r)[-20:][::-1].tolist()})

    t0 = time.time(); a,h = cpu_hits(A); t_h = time.time()-t0
    report["cpu"].append({"algo": "hits", "seconds": t_h,
                          "top20_authority": np.argsort(a)[-20:][::-1].tolist(),
                          "top20_hub": np.argsort(h)[-20:][::-1].tolist()})

    t0 = time.time(); dist = cpu_bfs(A, 0); t_b = time.time()-t0
    report["cpu"].append({"algo": "bfs", "seconds": t_b, "reachable": int((dist >= 0).sum())})

    # Compare top-20 overlap (simple correctness signal)
    def top20(run): 
        if "top20" in run: return set(run["top20"])
        return None

    cpu_map = {r["algo"]: r for r in report["cpu"]}
    gpu_map = {r["algo"]: r for r in report["gpu"]}

    for algo in ["pagerank", "rwr"]:
        if algo in cpu_map and algo in gpu_map:
            c = set(cpu_map[algo]["top20"])
            g = set(gpu_map[algo]["top20"])
            j = len(c & g) / max(1, len(c | g))
            speedup = cpu_map[algo]["seconds"] / max(1e-9, gpu_map[algo]["seconds"])
            report["comparisons"].append({"algo": algo, "top20_jaccard": j, "speedup": speedup})

    # BFS reachable compare
    if "bfs" in cpu_map and "bfs" in gpu_map:
        speedup = cpu_map["bfs"]["seconds"] / max(1e-9, gpu_map["bfs"]["seconds"])
        report["comparisons"].append({"algo": "bfs", "reachable_cpu": cpu_map["bfs"]["reachable"],
                                      "reachable_gpu": gpu_map["bfs"]["reachable"], "speedup": speedup})

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved:", out_json)
    print("Comparisons:")
    for c in report["comparisons"]:
        print(" ", c)

if __name__ == "__main__":
    main()