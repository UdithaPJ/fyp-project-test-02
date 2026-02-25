import os, sys, json, time
import numpy as np
from tqdm import tqdm
from scipy import sparse as sp

# ---------- helpers ----------
def load_meta(out_dir: str):
    meta_path = os.path.join(out_dir, "preprocess_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    shape = meta["shape"]  # expects [n_samples, n_features] from your previous stage
    mmap_path = meta["mmap_path"]
    if len(shape) != 2:
        raise RuntimeError(f"Unexpected shape in meta: {shape}")

    n_samples, n_features = int(shape[0]), int(shape[1])
    return meta, mmap_path, n_samples, n_features

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- GPU top-k updater ----------
def _update_topk_gpu(cp, topv, topi, candv, candi, k):
    """
    topv/topi: (B,k) current best
    candv/candi: (B,C) candidate values/indices
    returns updated (B,k)
    """
    B = topv.shape[0]
    # concat: (B, k+C)
    allv = cp.concatenate([topv, candv], axis=1)
    alli = cp.concatenate([topi, candi], axis=1)

    # take top-k by value
    idx = cp.argpartition(allv, -k, axis=1)[:, -k:]             # (B,k) unsorted top-k indices
    vals = cp.take_along_axis(allv, idx, axis=1)
    inds = cp.take_along_axis(alli, idx, axis=1)

    # sort descending
    order = cp.argsort(vals, axis=1)[:, ::-1]
    vals = cp.take_along_axis(vals, order, axis=1)
    inds = cp.take_along_axis(inds, order, axis=1)
    return vals, inds

# ---------- main ----------
def build_feature_topk_graph_gpu(
    out_dir="out",
    k=50,
    workers_note="gpu",
    # controls for memory/performance
    q_feat_block=128,      # query feature block size (rows for which we find neighbors)
    c_feat_block=1024,     # candidate feature block size (scanned across all features)
    sample_block=256,      # samples per streaming chunk from memmap
    symmetric=True,
    dtype=np.float32,
):
    import cupy as cp
    import cupyx.scipy.sparse as csp

    meta, mmap_path, n_samples, n_features = load_meta(out_dir)
    print("=== GPU top-k feature graph build ===")
    print("mmap_path:", mmap_path)
    print("shape(samples x features):", (n_samples, n_features))
    print(f"k={k} q_feat_block={q_feat_block} c_feat_block={c_feat_block} sample_block={sample_block} symmetric={symmetric}")

    # mmap view (CPU)
    X = np.memmap(mmap_path, mode="r", dtype=dtype, shape=(n_samples, n_features))

    # ---------- norms on GPU (streamed) ----------
    t_total0 = time.time()
    t0 = time.time()
    norms2 = cp.zeros(n_features, dtype=cp.float32)

    for s0 in tqdm(range(0, n_samples, sample_block), desc="gpu_norms"):
        s1 = min(n_samples, s0 + sample_block)
        chunk = cp.asarray(np.asarray(X[s0:s1, :], dtype=np.float32))  # (sb, nf)
        norms2 += cp.sum(chunk * chunk, axis=0)

    inv_norms = 1.0 / (cp.sqrt(norms2) + 1e-12)
    cp.cuda.Device().synchronize()
    t_norms = time.time() - t0
    print(f"Norms computed: {t_norms:.2f}s")

    # ---------- compute top-k on GPU in blocks ----------
    t0 = time.time()

    # We’ll collect edges in CPU lists (small enough: ~n*k)
    rows_cpu = []
    cols_cpu = []
    vals_cpu = []

    # query blocks
    for f0 in tqdm(range(0, n_features, q_feat_block), desc="query_blocks"):
        f1 = min(n_features, f0 + q_feat_block)
        B = f1 - f0

        # init top-k containers on GPU
        topv = cp.full((B, k), -cp.inf, dtype=cp.float32)
        topi = cp.full((B, k), -1, dtype=cp.int32)

        inv_q = inv_norms[f0:f1]  # (B,)

        # scan candidate feature blocks
        for c0 in range(0, n_features, c_feat_block):
            c1 = min(n_features, c0 + c_feat_block)
            C = c1 - c0

            # accumulate dot products: (B, C)
            sim = cp.zeros((B, C), dtype=cp.float32)

            # stream samples
            for s0 in range(0, n_samples, sample_block):
                s1 = min(n_samples, s0 + sample_block)
                # load only needed slices from disk
                Xq = cp.asarray(np.asarray(X[s0:s1, f0:f1], dtype=np.float32))  # (sb,B)
                Xc = cp.asarray(np.asarray(X[s0:s1, c0:c1], dtype=np.float32))  # (sb,C)
                sim += (Xq.T @ Xc)  # (B,C)

            # cosine normalize
            sim *= inv_q[:, None]
            sim *= inv_norms[c0:c1][None, :]

            # exclude self if overlap
            if (c0 <= f1) and (c1 >= f0):
                # for each query feature fi, if it's inside [c0,c1), set sim row col to -inf
                for bi in range(B):
                    fi = f0 + bi
                    if c0 <= fi < c1:
                        sim[bi, fi - c0] = -cp.inf

            # build candidate indices matrix (B,C) cheaply
            cand_idx = cp.arange(c0, c1, dtype=cp.int32)[None, :].repeat(B, axis=0)

            # update top-k
            topv, topi = _update_topk_gpu(cp, topv, topi, sim, cand_idx, k)

            # free some memory
            del sim, cand_idx
            cp._default_memory_pool.free_all_blocks()

        # move this query block results back to CPU
        rows = np.repeat(np.arange(f0, f1, dtype=np.int32), k)
        cols = cp.asnumpy(topi.reshape(-1)).astype(np.int32, copy=False)
        vals = cp.asnumpy(topv.reshape(-1)).astype(np.float32, copy=False)

        rows_cpu.append(rows)
        cols_cpu.append(cols)
        vals_cpu.append(vals)

        # clear GPU block memory
        del topv, topi
        cp._default_memory_pool.free_all_blocks()

    cp.cuda.Device().synchronize()
    t_blocks = time.time() - t0
    print(f"Top-k compute done: {t_blocks:.2f}s")

    # ---------- build CSR on GPU then save ----------
    t0 = time.time()
    rows = np.concatenate(rows_cpu)
    cols = np.concatenate(cols_cpu)
    vals = np.concatenate(vals_cpu)

    # symmetric (optional)
    if symmetric:
        rows2 = np.concatenate([rows, cols]).astype(np.int32, copy=False)
        cols2 = np.concatenate([cols, rows]).astype(np.int32, copy=False)
        vals2 = np.concatenate([vals, vals]).astype(np.float32, copy=False)
    else:
        rows2, cols2, vals2 = rows, cols, vals

    # COO -> CSR on GPU
    import cupy as cp
    import cupyx.scipy.sparse as csp

    Rg = cp.asarray(rows2)
    Cg = cp.asarray(cols2)
    Vg = cp.asarray(vals2)

    Gg = csp.coo_matrix((Vg, (Rg, Cg)), shape=(n_features, n_features)).tocsr()
    Gg.sum_duplicates()
    cp.cuda.Device().synchronize()

    # bring CSR to CPU for saving with scipy
    indptr = cp.asnumpy(Gg.indptr).astype(np.int32, copy=False)
    indices = cp.asnumpy(Gg.indices).astype(np.int32, copy=False)
    data = cp.asnumpy(Gg.data).astype(np.float32, copy=False)

    G = sp.csr_matrix((data, indices, indptr), shape=(n_features, n_features))
    out_path = os.path.join(out_dir, f"feature_graph_top{k}_csr_gpu.npz")
    sp.save_npz(out_path, G)

    t_save = time.time() - t0
    t_total = time.time() - t_total0

    print("Saved:", out_path)
    print("nnz:", G.nnz)
    print(f"Timing: norms={t_norms:.2f}s topk={t_blocks:.2f}s csr_build+save={t_save:.2f}s total={t_total:.2f}s")

    timing = {
        "stage": "csr_build",
        "mode": "gpu",
        "k": int(k),
        "q_feat_block": int(q_feat_block),
        "c_feat_block": int(c_feat_block),
        "sample_block": int(sample_block),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "nnz": int(G.nnz),
        "seconds_norms": float(t_norms),
        "seconds_topk_compute": float(t_blocks),
        "seconds_csr_build_save": float(t_save),
        "seconds_total": float(t_total),
        "output_graph": os.path.abspath(out_path),
    }
    with open(os.path.join(out_dir, "graph_build_gpu.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/03c_build_feature_topk_graph_gpu.py <out_dir> [k] [q_feat_block] [c_feat_block] [sample_block]")
        print("Example:")
        print("  python scripts/03c_build_feature_topk_graph_gpu.py out 50 128 1024 256")
        raise SystemExit(2)

    out_dir = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 50
    q_feat_block = int(sys.argv[3]) if len(sys.argv) >= 4 else 128
    c_feat_block = int(sys.argv[4]) if len(sys.argv) >= 5 else 1024
    sample_block = int(sys.argv[5]) if len(sys.argv) >= 6 else 256

    build_feature_topk_graph_gpu(
        out_dir=out_dir,
        k=k,
        q_feat_block=q_feat_block,
        c_feat_block=c_feat_block,
        sample_block=sample_block,
        symmetric=True,
    )

if __name__ == "__main__":
    main()