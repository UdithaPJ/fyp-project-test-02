import os
import sys
import json
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def pick_id_column(cols, sample_cols, schema):
    candidates = ["Name", "Description", "gene_id", "feature_id", "id", "index"]
    for c in candidates:
        if c in cols and c not in sample_cols:
            return c
    for f in schema:
        t = str(f.type).lower()
        if f.name not in sample_cols and ("string" in t):
            return f.name
    for c in cols:
        if c not in sample_cols:
            return c
    return None

def rb_block_to_numpy_2d(rb: pa.RecordBatch, col_start: int, col_end: int, dtype=np.float32):
    """
    Convert a slice of columns [col_start:col_end] (0-based within rb)
    into a 2D numpy array of shape (rows, block_cols) efficiently.

    Assumes those columns are numeric.
    """
    # Build a Table from the column slice (avoids Python per-column conversion loops)
    sub = pa.Table.from_batches([rb]).select(rb.schema.names[col_start:col_end])

    # Convert to numpy with minimal overhead
    # This returns (rows, cols)
    arr = sub.to_pandas(types_mapper=None).to_numpy(dtype=dtype, copy=False)
    return arr

def main(
    parquet_path: str,
    out_dir: str = "out",
    batch_rows: int = 1024,        # bigger batches reduce overhead (still safe with 12GB)
    sample_block_cols: int = 256,  # smaller block to keep conversion manageable
    flush_every: int = 200,        # flush less often to reduce I/O stalls
    dtype=np.float32,
):
    ensure_dir(out_dir)

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    cols = [f.name for f in schema]

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

    est_gb = (n_features * n_samples * np.dtype(dtype).itemsize) / (1024**3)
    print(f"\nEstimated memmap size: ~{est_gb:.1f} GB (make sure disk has plenty of free space!)")

    # Save sample IDs
    with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as f:
        for s in sample_cols:
            f.write(s + "\n")

    # features x samples for contiguous writes
    mmap_path = os.path.join(out_dir, "X_features_x_samples.float32.mmap")
    X = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=(n_features, n_samples))

    # Feature IDs file
    feat_path = os.path.join(out_dir, "features.txt")
    feat_file = open(feat_path, "w", encoding="utf-8")

    row_offset = 0
    batch_counter = 0

    print("\nStreaming parquet -> memmap (features x samples, vectorized conversion) ...")
    t_start = time.time()

    for rg in range(n_row_groups):
        it = pf.iter_batches(
            row_groups=[rg],
            batch_size=batch_rows,
            columns=[id_col] + sample_cols
        )

        for rb in tqdm(it, desc=f"row_group {rg}"):
            # rb is a RecordBatch: col0=id, col1..=samples
            b_rows = rb.num_rows
            r0 = row_offset
            r1 = row_offset + b_rows

            # write feature ids
            ids = rb.column(0).to_pylist()
            feat_file.write("\n".join(map(str, ids)) + "\n")

            # write numeric data in blocks, but convert each block vectorized
            # sample columns start at 1 in the RecordBatch
            for c0 in range(0, n_samples, sample_block_cols):
                c1 = min(n_samples, c0 + sample_block_cols)

                # record batch column indices: [1+c0 : 1+c1]
                M = rb_block_to_numpy_2d(rb, 1 + c0, 1 + c1, dtype=dtype)  # (rows, block_cols)
                X[r0:r1, c0:c1] = M

            row_offset = r1
            batch_counter += 1

            if batch_counter % flush_every == 0:
                X.flush()

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

    print(f"\nDone in {time.time() - t_start:.1f}s")
    print("Saved:")
    print(" ", mmap_path)
    print(" ", feat_path)
    print(" ", os.path.join(out_dir, "samples.txt"))
    print(" ", os.path.join(out_dir, "preprocess_meta.json"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/02_wideparquet_to_memmap_fast2.py <parquet_path> [out_dir]")
        raise SystemExit(2)
    parquet = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "out"
    main(parquet, out_dir=out_dir)