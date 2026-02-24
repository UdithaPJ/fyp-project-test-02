import os, sys, json
import numpy as np
from tqdm import tqdm
from scipy import sparse

def main(out_dir="out", k=50, feat_block=512, sample_block=512, use_gpu=False):
    meta = json.load(open(os.path.join(out_dir, "preprocess_meta.json"), "r", encoding="utf-8"))
    n_samples, n_features = meta["shape"]
    X = np.memmap(meta["mmap_path"], mode="r", dtype=np.float32, shape=(n_samples, n_features))

    xp = np
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            print("Using GPU via CuPy for block matmul.")
        except Exception as e:
            print("CuPy not available; using CPU.", e)
            use_gpu = False
            xp = np

    # Norms of features (columns)
    print("Computing feature norms...")
    norms = np.zeros(n_features, dtype=np.float64)
    for s0 in tqdm(range(0, n_samples, sample_block)):
        s1 = min(n_samples, s0 + sample_block)
        B = np.asarray(X[s0:s1, :], dtype=np.float32)
        norms += (B.astype(np.float64) ** 2).sum(axis=0)
    norms = np.sqrt(norms) + 1e-12
    inv = (1.0 / norms).astype(np.float32)

    rows_all, cols_all, vals_all = [], [], []

    print("Building top-k cosine graph (feature blocks)...")
    for f0 in tqdm(range(0, n_features, feat_block)):
        f1 = min(n_features, f0 + feat_block)
        bsz = f1 - f0
        scores = np.zeros((bsz, n_features), dtype=np.float32)

        for s0 in range(0, n_samples, sample_block):
            s1 = min(n_samples, s0 + sample_block)
            chunk = np.asarray(X[s0:s1, :], dtype=np.float32)

            A = chunk[:, f0:f1]  # (s_block x bsz)
            B = chunk            # (s_block x n_features)

            if use_gpu:
                import cupy as cp
                Ag = cp.asarray(A)
                Bg = cp.asarray(B)
                scores += cp.asnumpy(Ag.T @ Bg).astype(np.float32)
                del Ag, Bg
                cp._default_memory_pool.free_all_blocks()
            else:
                scores += (A.T @ B).astype(np.float32)

        # cosine normalize
        scores *= inv[f0:f1, None]
        scores *= inv[None, :]

        for bi in range(bsz):
            fi = f0 + bi
            scores[bi, fi] = -np.inf
            idx = np.argpartition(scores[bi], -k)[-k:]
            idx = idx[np.argsort(scores[bi, idx])[::-1]]
            val = scores[bi, idx]

            rows_all.append(np.full(k, fi, dtype=np.int32))
            cols_all.append(idx.astype(np.int32))
            vals_all.append(val.astype(np.float32))

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all)

    # Symmetrize
    R = np.concatenate([rows, cols])
    C = np.concatenate([cols, rows])
    V = np.concatenate([vals, vals])

    G = sparse.coo_matrix((V, (R, C)), shape=(n_features, n_features)).tocsr()
    G.sum_duplicates()

    out_path = os.path.join(out_dir, f"feature_graph_top{k}_csr.npz")
    sparse.save_npz(out_path, G)
    print("Saved:", out_path, "nnz=", G.nnz)

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else "out"
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 50
    use_gpu = (sys.argv[3].lower() == "gpu") if len(sys.argv) >= 4 else False
    main(out_dir=out_dir, k=k, use_gpu=use_gpu)