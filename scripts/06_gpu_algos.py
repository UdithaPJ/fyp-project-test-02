import os, sys, json, time
import numpy as np
from scipy import sparse as sp

def load_csr_npz(path: str):
    A = sp.load_npz(path).tocsr()
    # Use float32 weights on GPU
    if A.dtype != np.float32:
        A = A.astype(np.float32)
    # Indices/indptr as int32 (better for GPU)
    if A.indices.dtype != np.int32:
        A.indices = A.indices.astype(np.int32)
    if A.indptr.dtype != np.int32:
        A.indptr = A.indptr.astype(np.int32)
    return A

def to_gpu_csr(A_csr):
    import cupy as cp
    import cupyx.scipy.sparse as csp

    indptr = cp.asarray(A_csr.indptr)
    indices = cp.asarray(A_csr.indices)
    data = cp.asarray(A_csr.data)
    n = A_csr.shape[0]
    G = csp.csr_matrix((data, indices, indptr), shape=(n, n))
    return G

def gpu_pagerank(G, d=0.85, tol=1e-6, max_iter=200):
    import cupy as cp
    import cupyx.scipy.sparse as csp

    n = G.shape[0]
    out = cp.asarray(G.sum(axis=1)).ravel()
    inv = cp.zeros_like(out, dtype=cp.float32)
    inv = cp.where(out > 0, 1.0 / out, 0.0).astype(cp.float32)
    P = csp.diags(inv) @ G  # row-stochastic

    pr = cp.full(n, 1.0 / n, dtype=cp.float32)
    teleport = cp.float32((1.0 - d) / n)

    for _ in range(max_iter):
        pr_new = teleport + cp.float32(d) * (P.T @ pr)
        err = cp.abs(pr_new - pr).sum()
        pr = pr_new
        if float(err.get()) < tol:
            break
    return pr

def gpu_rwr(G, seed=0, restart=0.5, tol=1e-6, max_iter=300):
    import cupy as cp
    import cupyx.scipy.sparse as csp

    n = G.shape[0]
    out = cp.asarray(G.sum(axis=1)).ravel()
    inv = cp.where(out > 0, 1.0 / out, 0.0).astype(cp.float32)
    P = csp.diags(inv) @ G

    p0 = cp.zeros(n, dtype=cp.float32)
    p0[int(seed)] = 1.0
    p = p0.copy()

    r = cp.float32(restart)
    one_minus = cp.float32(1.0 - restart)

    for _ in range(max_iter):
        p_new = one_minus * (P.T @ p) + r * p0
        err = cp.abs(p_new - p).sum()
        p = p_new
        if float(err.get()) < tol:
            break
    return p

def gpu_hits(G, tol=1e-6, max_iter=300):
    import cupy as cp

    n = G.shape[0]
    h = cp.ones(n, dtype=cp.float32)
    a = cp.ones(n, dtype=cp.float32)

    for _ in range(max_iter):
        a_new = G.T @ h
        h_new = G @ a_new
        a_new = a_new / (cp.linalg.norm(a_new) + 1e-12)
        h_new = h_new / (cp.linalg.norm(h_new) + 1e-12)

        err = cp.linalg.norm(a_new - a) + cp.linalg.norm(h_new - h)
        a, h = a_new, h_new
        if float(err.get()) < tol:
            break
    return a, h

def gpu_bfs(G, source=0, max_depth=1000000):
    """
    Frontier-based BFS using CSR adjacency on GPU.
    This uses boolean frontier propagation: frontier_next = (A * frontier) > 0
    Efficient enough for sparse graphs; good for milestone 2.1 verification.
    """
    import cupy as cp
    import cupyx.scipy.sparse as csp

    n = G.shape[0]
    src = int(source)

    visited = cp.zeros(n, dtype=cp.bool_)
    frontier = cp.zeros(n, dtype=cp.bool_)
    frontier[src] = True
    visited[src] = True

    dist = cp.full(n, -1, dtype=cp.int32)
    dist[src] = 0

    depth = 0
    # Convert boolean frontier to float32 vector for sparse matvec
    while bool(frontier.any().get()):
        if depth >= max_depth:
            break
        # neighbors = G.T @ frontier_vec  (use outgoing edges; either is okay if symmetric)
        frontier_vec = frontier.astype(cp.float32)
        neigh = (G.T @ frontier_vec)  # float
        next_frontier = neigh > 0
        next_frontier = cp.logical_and(next_frontier, ~visited)

        depth += 1
        dist[next_frontier] = depth
        visited = cp.logical_or(visited, next_frontier)
        frontier = next_frontier

    return dist

def gpu_mean_conditional_entropy(G, labels):
    """
    Mean Conditional Entropy of neighbor label distribution:
      For each node i: H(L_neighbor | i) = -sum_c p_ic log(p_ic)
      Return mean over nodes with degree>0.
    labels: numpy int array shape (n,)
    """
    import cupy as cp

    n = G.shape[0]
    lab = cp.asarray(labels, dtype=cp.int32)

    # Edge list from CSR
    indptr = G.indptr
    indices = G.indices

    # degrees
    deg = (indptr[1:] - indptr[:-1]).astype(cp.int32)

    # Build edge arrays (u,v) expanded
    # (This expansion is O(nnz) and fits in VRAM for your graph)
    u = cp.repeat(cp.arange(n, dtype=cp.int32), deg)
    v = indices.astype(cp.int32)

    lv = lab[v]
    # Map labels to compact range [0..C-1]
    uniq, inv = cp.unique(lv, return_inverse=True)
    C = int(uniq.shape[0])

    # Count per (u,label)
    # key = u*C + inv_label
    key = u.astype(cp.int64) * C + inv.astype(cp.int64)
    counts = cp.bincount(key, minlength=n * C).reshape(n, C).astype(cp.float32)

    deg_f = deg.astype(cp.float32)
    # Avoid division by zero
    mask = deg_f > 0
    p = cp.zeros_like(counts)
    p[mask] = counts[mask] / deg_f[mask, None]

    # entropy per node
    # 0*log(0) treated as 0
    plogp = cp.where(p > 0, p * cp.log(p), 0.0)
    H = -cp.sum(plogp, axis=1)
    mean_H = cp.mean(H[mask]) if bool(mask.any().get()) else cp.float32(0.0)
    return mean_H

def gpu_time(fn, *args, warmup=1, iters=1):
    import cupy as cp
    # warmup
    for _ in range(warmup):
        out = fn(*args)
        cp.cuda.Device().synchronize()
        del out

    start = cp.cuda.Event(); end = cp.cuda.Event()
    start.record()
    out = None
    for _ in range(iters):
        out = fn(*args)
    end.record()
    end.synchronize()
    ms = cp.cuda.get_elapsed_time(start, end)
    return out, ms / 1000.0

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python scripts/06_gpu_algos.py <graph_npz> <out_dir>")
        print("Example:")
        print("  python scripts/06_gpu_algos.py out/feature_graph_top50_csr_parallel_w4.npz out/gpu_results")
        raise SystemExit(2)

    graph_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    A = load_csr_npz(graph_path)
    n = A.shape[0]
    print("Loaded CSR:", A.shape, "nnz=", A.nnz)

    # GPU transfer
    import cupy as cp
    cp.cuda.Device().use()
    G = to_gpu_csr(A)
    cp.cuda.Device().synchronize()
    print("Transferred to GPU.")

    results = {"graph_path": os.path.abspath(graph_path), "n": int(n), "nnz": int(A.nnz), "gpu": True, "runs": []}

    # PageRank
    pr, t_pr = gpu_time(gpu_pagerank, G, 0.85, 1e-6, 200, warmup=1, iters=1)
    top_pr = cp.asnumpy(cp.argsort(pr)[-20:][::-1]).tolist()
    results["runs"].append({"algo": "pagerank", "seconds": t_pr, "top20": top_pr})

    # RWR
    rwr, t_rwr = gpu_time(gpu_rwr, G, 0, 0.5, 1e-6, 300, warmup=1, iters=1)
    top_rwr = cp.asnumpy(cp.argsort(rwr)[-20:][::-1]).tolist()
    results["runs"].append({"algo": "rwr", "seconds": t_rwr, "top20": top_rwr})

    # HITS
    (a, h), t_hits = gpu_time(gpu_hits, G, 1e-6, 300, warmup=1, iters=1)
    top_a = cp.asnumpy(cp.argsort(a)[-20:][::-1]).tolist()
    top_h = cp.asnumpy(cp.argsort(h)[-20:][::-1]).tolist()
    results["runs"].append({"algo": "hits", "seconds": t_hits, "top20_authority": top_a, "top20_hub": top_h})

    # BFS
    dist, t_bfs = gpu_time(gpu_bfs, G, 0, 1000000, warmup=1, iters=1)
    reachable = int(cp.sum(dist >= 0).get())
    results["runs"].append({"algo": "bfs", "seconds": t_bfs, "reachable": reachable})

    # Mean Conditional Entropy demo: random labels (replace with Louvain labels later)
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 50, size=n, dtype=np.int32)
    mce, t_mce = gpu_time(gpu_mean_conditional_entropy, G, labels, warmup=1, iters=1)
    results["runs"].append({"algo": "mean_conditional_entropy", "seconds": t_mce, "value": float(mce.get())})

    out_json = os.path.join(out_dir, "gpu_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved:", out_json)

if __name__ == "__main__":
    main()