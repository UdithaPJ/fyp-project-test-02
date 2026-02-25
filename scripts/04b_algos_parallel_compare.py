import os, sys, json, time
import numpy as np
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits

def pagerank_csr(A, d=0.85, tol=1e-6, max_iter=100):
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

def rwr_csr(A, seed=0, restart=0.5, tol=1e-6, max_iter=200):
    n = A.shape[0]
    out = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(out, dtype=np.float64)
    inv[out > 0] = 1.0 / out[out > 0]
    P = (sparse.diags(inv) @ A).tocsr()

    p0 = np.zeros(n, dtype=np.float64)
    p0[seed] = 1.0
    p = p0.copy()
    for _ in range(max_iter):
        p_new = (1.0 - restart) * (P.T @ p) + restart * p0
        if np.abs(p_new - p).sum() < tol:
            p = p_new
            break
        p = p_new
    return p

def hits_csr(A, tol=1e-6, max_iter=200):
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

def _run_algo(algo, graph_path, out_dir):
    with threadpool_limits(limits=1):
        t_load0 = time.time()
        A = sparse.load_npz(graph_path).tocsr()
        t_load = time.time() - t_load0

        t0 = time.time()
        if algo == "pagerank":
            pr = pagerank_csr(A)
            t_comp = time.time() - t0
            res = {"algo": algo, "seconds_load_graph": t_load, "seconds_compute": t_comp,
                   "seconds_total": t_load + t_comp,
                   "top20": np.argsort(pr)[-20:][::-1].tolist()}
        elif algo == "rwr":
            r = rwr_csr(A, seed=0)
            t_comp = time.time() - t0
            res = {"algo": algo, "seconds_load_graph": t_load, "seconds_compute": t_comp,
                   "seconds_total": t_load + t_comp,
                   "top20": np.argsort(r)[-20:][::-1].tolist()}
        elif algo == "hits":
            a, h = hits_csr(A)
            t_comp = time.time() - t0
            res = {
                "algo": algo,
                "seconds_load_graph": t_load,
                "seconds_compute": t_comp,
                "seconds_total": t_load + t_comp,
                "top20_authority": np.argsort(a)[-20:][::-1].tolist(),
                "top20_hub": np.argsort(h)[-20:][::-1].tolist()
            }
        else:
            raise ValueError(algo)

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{algo}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        return res

def run_compare(graph_path, mode="single"):
    algos = ["pagerank", "rwr", "hits"]
    out_dir = os.path.join("out", "algo_results", mode)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    if mode == "single":
        results = [_run_algo(a, graph_path, out_dir) for a in algos]
    else:
        with ProcessPoolExecutor(max_workers=len(algos)) as ex:
            futures = [ex.submit(_run_algo, a, graph_path, out_dir) for a in algos]
            results = [f.result() for f in futures]
    total_wall = time.time() - t0

    summary = {
        "mode": mode,
        "graph_path": os.path.abspath(graph_path),
        "total_seconds_wall": float(total_wall),
        "results": results
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    graph_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) >= 3 else "single"
    run_compare(graph_path, mode=mode)