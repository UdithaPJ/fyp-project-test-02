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
    n_samples, n_features = meta["shape"]
    mmap_path = meta["mmap_path"]
    return meta, mmap_path, n_samples, n_features

def compute_feature_inv_norms(mmap_path: str, n_samples: int, n_features: int, sample_block: int):
    X = np.memmap(mmap_path, mode="r", dtype=np.float32, shape=(n_samples, n_features))
    norms = np.zeros(n_features, dtype=np.float64)
    for s0 in tqdm(range(0, n_samples, sample_block), desc="norms"):
        s1 = min(n_samples, s0 + sample_block)
        chunk = np.asarray(X[s0:s1, :], dtype=np.float32)
        norms += (chunk.astype(np.float64) ** 2).sum(axis=0)
    norms = np.sqrt(norms) + 1e-12
    return (1.0 / norms).astype(np.float32)

def _worker_block(args):
    out_dir, mmap_path, n_samples, n_features, inv_norms, k, f0, f1, sample_block = args
    with threadpool_limits(limits=1):
        X = np.memmap(mmap_path, mode="r", dtype=np.float32, shape=(n_samples, n_features))
        bsz = f1 - f0
        scores = np.zeros((bsz, n_features), dtype=np.float32)

        for s0 in range(0, n_samples, sample_block):
            s1 = min(n_samples, s0 + sample_block)
            chunk = np.asarray(X[s0:s1, :], dtype=np.float32)
            A = chunk[:, f0:f1]
            scores += (A.T @ chunk).astype(np.float32)

        scores *= inv_norms[f0:f1, None]
        scores *= inv_norms[None, :]

        rows = np.empty(bsz * k, dtype=np.int32)
        cols = np.empty(bsz * k, dtype=np.int32)
        vals = np.empty(bsz * k, dtype=np.float32)

        for bi in range(bsz):
            fi = f0 + bi
            scores[bi, fi] = -np.inf
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

    t_total0 = time.time()

    # Stage 1: norms
    t0 = time.time()
    inv_norms = compute_feature_inv_norms(mmap_path, n_samples, n_features, sample_block=sample_block)
    t_norms = time.time() - t0

    # Stage 2: block compute (parallel)
    tasks = []
    for f0 in range(0, n_features, feat_block):
        f1 = min(n_features, f0 + feat_block)
        tasks.append((out_dir, mmap_path, n_samples, n_features, inv_norms, k, f0, f1, sample_block))

    t0 = time.time()
    chunk_files = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker_block, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="blocks"):
            chunk_files.append(fut.result())
    t_blocks = time.time() - t0

    # Stage 3: merge + CSR + save
    t0 = time.time()
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
    t_merge = time.time() - t0

    t_total = time.time() - t_total0

    print("Saved:", out_path)
    print("nnz:", G.nnz)
    print(f"Timing: norms={t_norms:.2f}s blocks={t_blocks:.2f}s merge/save={t_merge:.2f}s total={t_total:.2f}s")

    timing = {
        "stage": "csr_build",
        "mode": "parallel" if workers > 1 else "single",
        "workers": int(workers),
        "k": int(k),
        "feat_block": int(feat_block),
        "sample_block": int(sample_block),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "nnz": int(G.nnz),
        "seconds_norms": float(t_norms),
        "seconds_blocks": float(t_blocks),
        "seconds_merge_save": float(t_merge),
        "seconds_total": float(t_total),
        "output_graph": os.path.abspath(out_path),
    }
    with open(os.path.join(out_dir, f"graph_build_parallel_w{workers}.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else "out"
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 50
    workers = int(sys.argv[3]) if len(sys.argv) >= 4 else 4
    feat_block = int(sys.argv[4]) if len(sys.argv) >= 5 else 128
    sample_block = int(sys.argv[5]) if len(sys.argv) >= 6 else 256
    build_graph_parallel(out_dir=out_dir, k=k, workers=workers, feat_block=feat_block, sample_block=sample_block)