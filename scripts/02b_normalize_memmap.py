import os, sys, json
import numpy as np
from tqdm import tqdm

def main(out_dir="out", row_chunk=256):
    meta_path = os.path.join(out_dir, "preprocess_meta.json")
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    n_samples, n_features = meta["shape"]
    mmap_path = meta["mmap_path"]

    X = np.memmap(mmap_path, mode="r+", dtype=np.float32, shape=(n_samples, n_features))

    # Library size per sample (sum across features)
    print("Computing library sizes (chunked)...")
    lib = np.zeros(n_samples, dtype=np.float64)
    for i0 in tqdm(range(0, n_samples, row_chunk)):
        i1 = min(n_samples, i0 + row_chunk)
        block = np.asarray(X[i0:i1, :], dtype=np.float32)
        lib[i0:i1] = block.sum(axis=1)

    np.save(os.path.join(out_dir, "libsize.npy"), lib)
    safe = np.where(lib > 0, lib, 1.0)
    scale = (1e6 / safe).astype(np.float32)

    # CPM-like scaling + log1p, in-place
    print("Normalizing in-place (CPM + log1p)...")
    for i0 in tqdm(range(0, n_samples, row_chunk)):
        i1 = min(n_samples, i0 + row_chunk)
        block = np.asarray(X[i0:i1, :], dtype=np.float32)
        block *= scale[i0:i1, None]
        np.log1p(block, out=block)
        X[i0:i1, :] = block
        X.flush()

    meta["normalization"] = "CPM-like (1e6*counts/libsize) then log1p"
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)

    print("Done. Saved libsize.npy and updated preprocess_meta.json")

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else "out"
    main(out_dir=out_dir)