import os
import sys
import json
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def pick_id_column(cols, sample_cols, schema):
    # Prefer a single string column that is NOT a GTEX sample column
    candidates = ["Name", "Description", "gene_id", "feature_id", "id", "index"]
    for c in candidates:
        if c in cols and c not in sample_cols:
            return c

    # Heuristic: pick first string-like field not in sample_cols
    for f in schema:
        t = str(f.type).lower()
        if f.name not in sample_cols and ("string" in t):
            return f.name

    # Fallback: first non-sample col
    for c in cols:
        if c not in sample_cols:
            return c
    return None

def main(
    parquet_path: str,
    out_dir: str = "out",
    batch_rows: int = 256,          # smaller batch keeps RAM low
    sample_block_cols: int = 512,   # number of sample columns stacked at once
    flush_every: int = 25,          # flush every N batches to reduce I/O stalls
    dtype=np.float32,
):
    ensure_dir(out_dir)

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    cols = [f.name for f in schema]

    # Detect sample columns by GTEX- prefix (fast + reliable for your files)
    sample_cols = [c for c in cols if c.startswith("GTEX-")]
    if not sample_cols:
        raise RuntimeError("No sample columns detected (expected columns starting with 'GTEX-').")

    id_col = pick_id_column(cols, sample_cols, schema)
    if id_col is None:
        raise RuntimeError("Could not detect feature ID column (e.g., Name/Description).")

    n_samples = len(sample_cols)
    n_features = pf.metadata.num_rows
    n_row_groups = pf.metadata.num_row_groups

    print(f"Row groups: {n_row_groups}, rows(features): {n_features}")
    print(f"Schema columns count: {len(cols)}")
    print("\nDetected wide format:")
    print("  id_col   :", id_col)
    print("  samples  :", n_samples)
    print("  features :", n_features)

    # Disk size estimate
    est_gb = (n_features * n_samples * np.dtype(dtype).itemsize) / (1024**3)
    print(f"\nEstimated memmap size: ~{est_gb:.1f} GB (ensure enough free disk space!)")

    # Save sample IDs
    with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as f:
        for s in sample_cols:
            f.write(s + "\n")

    # IMPORTANT: store as (features x samples) for contiguous writes
    mmap_path = os.path.join(out_dir, "X_features_x_samples.float32.mmap")
    X = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=(n_features, n_samples))
    X[:] = 0
    X.flush()

    # Feature IDs
    feat_path = os.path.join(out_dir, "features.txt")
    feat_file = open(feat_path, "w", encoding="utf-8")

    row_offset = 0
    batch_counter = 0

    print("\nStreaming parquet -> memmap (features x samples, column-blocked) ...")

    for rg in range(n_row_groups):
        it = pf.iter_batches(
            row_groups=[rg],
            batch_size=batch_rows,
            columns=[id_col] + sample_cols
        )

        for batch in tqdm(it, desc=f"row_group {rg}"):
            # Write feature ids
            arr_id = batch.column(0)
            ids = arr_id.to_pylist()
            for v in ids:
                feat_file.write(str(v) + "\n")

            b_rows = len(ids)
            r0 = row_offset
            r1 = row_offset + b_rows

            # Write sample values in blocks of columns to avoid huge temporary arrays
            # batch columns: 0 = id_col, 1..n_samples = sample cols
            for c0 in range(0, n_samples, sample_block_cols):
                c1 = min(n_samples, c0 + sample_block_cols)

                # Build a small matrix (b_rows x (c1-c0)) efficiently
                # We stack only this subset of columns
                cols_np = []
                # +1 because batch col 0 is id_col
                for j in range(c0, c1):
                    cols_np.append(np.asarray(batch.column(j + 1)))

                M = np.stack(cols_np, axis=1).astype(dtype, copy=False)  # (b_rows x block_cols)

                # Contiguous write into memmap
                X[r0:r1, c0:c1] = M

            row_offset = r1
            batch_counter += 1

            # Flush occasionally (not every batch)
            if batch_counter % flush_every == 0:
                X.flush()

    # Final flush/close
    X.flush()
    feat_file.close()

    if row_offset != n_features:
        raise RuntimeError(f"Wrote {row_offset} features but expected {n_features}.")

    meta = {
        "parquet_path": os.path.abspath(parquet_path),
        "shape": [int(n_features), int(n_samples)],
        "mmap_path": os.path.abspath(mmap_path),
        "format": "wide (samples are columns, first column is feature id)",
        "orientation": "features_by_samples",
        "id_col": id_col,
        "dtype": "float32",
    }
    with open(os.path.join(out_dir, "preprocess_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print("Saved:")
    print(" ", mmap_path)
    print(" ", feat_path)
    print(" ", os.path.join(out_dir, "samples.txt"))
    print(" ", os.path.join(out_dir, "preprocess_meta.json"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/02_wideparquet_to_memmap_fast.py <parquet_path> [out_dir]")
        raise SystemExit(2)
    parquet = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "out"
    main(parquet, out_dir=out_dir)