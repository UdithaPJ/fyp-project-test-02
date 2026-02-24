import os, sys, json, time
import numpy as np
from tqdm import tqdm
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from threadpoolctl import threadpool_limits

def load_meta(out_dir: str):
    meta_path = os.path.join(out_dir, "preprocess_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Your meta implies samples_by_features
    n_samples, n_features = meta["shape"]
    mmap_path = meta["mmap_path"]

    return meta, mmap_path, n_samples, n_features

def compute_feature_inv_norms(mmap_path: str, n_samples: int, n_features: int, sample_block: int):
    X = np.memmap(mmap_path, mode="r", dtype=np.float32, shape=(n_samples, n_features))
    norms = np.zeros(n_features, dtype=np.float64)

    for s0 in tqdm(range(0, n_samples, sample_block), desc="norms"):
        s1 = min(n_samples, s0 + sample_block)
        chunk = np.asarray(X[s0:s1, :], dtype=np.float32)  # (sb x nf)
        norms += (chunk.astype(np.float64) ** 2).sum(axis=0)

    norms = np.sqrt(norms) + 1e-12
    return (1.0 / norms).astype(np.float32)

def _worker_block(args):
    """
    Compute top-k cosine neighbors for feature block [f0, f1) against all features.
    Save edges to out/chunks/*.npz and return the path.
    """
    out_dir, mmap_path, n_samples, n_features, inv_norms, k, f0, f1, sample_block = args

    # Important: one BLAS thread per process to avoid oversubscription
    with threadpool_limits(limits=1):
        X = np.memmap(mmap_path, mode="r", dtype=np.float32, shape=(n_samples, n_features))
        bsz = f1 - f0

        scores = np.zeros((bsz, n_features), dtype=np.float32)

        # accumulate dot products block vs all
        for s0 in range(0, n_samples, sample_block):
            s1 = min(n_samples, s0 + sample_block)
            chunk = np.asarray(X[s0:s1, :], dtype=np.float32)     # (sb x nf)
            A = chunk[:, f0:f1]                                    # (sb x bsz)
            scores += (A.T @ chunk).astype(np.float32)             # (bsz x nf)

        # cosine normalize
        scores *= inv_norms[f0:f1, None]
        scores *= inv_norms[None, :]

        # top-k per row
        rows = np.empty(bsz * k, dtype=np.int32)
        cols = np.empty(bsz * k, dtype=np.int32)
        vals = np.empty(bsz * k, dtype=np.float32)

        for bi in range(bsz):
            fi = f0 + bi
            scores[bi, fi] = -np.inf  # exclude self

            idx = np.argpartition(scores[bi], -k)[-k:]
            idx = idx[np.argsort(scores[bi, idx])[::-1]]
            val = scores[bi, idx]

            o0 = bi * k
            o1 = o0 + k
            rows[o0:o1] = fi
            cols[o0:o1] = idx.astype(np.int32, copy=False)
            vals[o0:o1] = val.astype(np.float32, copy=False)

        os.makedirs(os.path.join(out_dir, "chunks"), exist_ok=True)
        chunk_path = os.path.join(out_dir, "chunks", f"edges_f{f0}_f{f1}_k{k}.npz")
        np.savez_compressed(chunk_path, rows=rows, cols=cols, vals=vals)

        return chunk_path

def build_graph_parallel(out_dir="out", k=50, workers=4, feat_block=128, sample_block=256, symmetric=True):
    meta, mmap_path, n_samples, n_features = load_meta(out_dir)

    print("=== Graph build (parallel) ===")
    print("mmap_path:", mmap_path)
    print("shape:", (n_samples, n_features))
    print(f"k={k} workers={workers} feat_block={feat_block} sample_block={sample_block} symmetric={symmetric}")

    t0 = time.time()
    inv_norms = compute_feature_inv_norms(mmap_path, n_samples, n_features, sample_block=sample_block)
    print(f"Computed norms in {time.time()-t0:.2f}s")

    # Create tasks
    tasks = []
    for f0 in range(0, n_features, feat_block):
        f1 = min(n_features, f0 + feat_block)
        tasks.append((out_dir, mmap_path, n_samples, n_features, inv_norms, k, f0, f1, sample_block))

    # Dispatch
    chunk_files = []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker_block, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="blocks"):
            chunk_files.append(fut.result())
    print(f"Computed blocks in {time.time()-t1:.2f}s")

    # Merge chunks
    print("Merging chunks -> CSR...")
    rows_all, cols_all, vals_all = [], [], []
    for p in tqdm(chunk_files, desc="read_chunks"):
        z = np.load(p)
        rows_all.append(z["rows"])
        cols_all.append(z["cols"])
        vals_all.append(z["vals"])

    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all)

    if symmetric:
        R = np.concatenate([rows, cols])
        C = np.concatenate([cols, rows])
        V = np.concatenate([vals, vals])
    else:
        R, C, V = rows, cols, vals

    G = sparse.coo_matrix((V, (R, C)), shape=(n_features, n_features)).tocsr()
    G.sum_duplicates()

    out_path = os.path.join(out_dir, f"feature_graph_top{k}_csr_parallel_w{workers}.npz")
    sparse.save_npz(out_path, G)

    total = time.time() - t0
    print("Saved:", out_path)
    print("nnz:", G.nnz)
    print(f"TOTAL graph build time: {total:.2f}s")

    # Save timing
    timing = {
        "mode": "parallel",
        "workers": workers,
        "k": k,
        "feat_block": feat_block,
        "sample_block": sample_block,
        "n_samples": n_samples,
        "n_features": n_features,
        "nnz": int(G.nnz),
        "seconds_total": float(total),
        "output_graph": os.path.abspath(out_path),
    }
    with open(os.path.join(out_dir, f"graph_build_parallel_w{workers}.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

def build_graph_single(out_dir="out", k=50, feat_block=128, sample_block=256, symmetric=True):
    # Same as parallel but workers=1 (fair baseline)
    build_graph_parallel(out_dir=out_dir, k=k, workers=1, feat_block=feat_block, sample_block=sample_block, symmetric=symmetric)

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else "out"
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 50
    workers = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
    feat_block = int(sys.argv[4]) if len(sys.argv) >= 5 else 128
    sample_block = int(sys.argv[5]) if len(sys.argv) >= 6 else 256

    build_graph_parallel(out_dir=out_dir, k=k, workers=workers, feat_block=feat_block, sample_block=sample_block)