import os, sys, time
import numpy as np
from scipy import sparse
from collections import deque

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

def bfs_csr(A, source=0, max_visits=200000):
    n = A.shape[0]
    dist = -np.ones(n, dtype=np.int32)
    q = deque([source])
    dist[source] = 0
    visited = 1
    indptr, indices = A.indptr, A.indices
    while q:
        u = q.popleft()
        for ei in range(indptr[u], indptr[u+1]):
            v = indices[ei]
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
                visited += 1
                if visited >= max_visits:
                    return dist
    return dist

def main(out_dir="out", k=50, topn=20):
    path = os.path.join(out_dir, f"feature_graph_top{k}_csr.npz")
    A = sparse.load_npz(path).tocsr()
    print("Graph:", A.shape, "nnz=", A.nnz)

    t0 = time.time()
    pr = pagerank_csr(A)
    print("PageRank:", time.time()-t0, "s")
    print("Top nodes:", np.argsort(pr)[-topn:][::-1])

    t0 = time.time()
    rw = rwr_csr(A, seed=0)
    print("RWR:", time.time()-t0, "s")
    print("Top nodes:", np.argsort(rw)[-topn:][::-1])

    t0 = time.time()
    dist = bfs_csr(A, source=0)
    print("BFS:", time.time()-t0, "s")
    print("Reachable:", int((dist >= 0).sum()), "/", A.shape[0])

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else "out"
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 50
    main(out_dir=out_dir, k=k)