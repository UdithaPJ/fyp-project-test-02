import os
import sys
import json
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def pick_id_column(all_cols, sample_cols):
    # common ID column names in GTEx-ish exports
    candidates = [
        "Name", "name",
        "gene_id", "gene", "gene_name", "Gene", "GENE",
        "feature_id", "Feature", "ID", "id",
        "Description", "desc",
        "index"
    ]
    for c in candidates:
        if c in all_cols and c not in sample_cols:
            return c

    # fallback: first column that is not a sample column
    for c in all_cols:
        if c not in sample_cols:
            return c

    return None

def main(
    parquet_path: str,
    out_dir: str = "out",
    dtype=np.float32,
    batch_rows: int = 1024,
):
    ensure_dir(out_dir)

    pf = pq.ParquetFile(parquet_path)
    schema_cols = [f.name for f in pf.schema_arrow]

    # sample columns are typically GTEX-... and are numeric; but we’ll infer by exclusion:
    # If there is exactly 1 string-ish column and the rest numeric, that’s the ID column.
    # Here we just start by assuming "samples are all columns except the ID column", but
    # we need ID col first. We'll detect it from the first batch.

    n_features = pf.metadata.num_rows
    print(f"Row groups: {pf.metadata.num_row_groups}, rows(features): {n_features}")
    print(f"Schema columns count: {len(schema_cols)}")

    # Read ONE small batch to detect actual column names and ID column
    it0 = pf.iter_batches(row_groups=[0], batch_size=min(batch_rows, 64))
    first_batch = next(iter(it0))
    bcols = first_batch.schema.names
    print("\nColumns seen in first batch (first 30):")
    print(bcols[:30])
    print("... total:", len(bcols))

    # Heuristic: sample columns are those that look like GTEX-* OR all numeric columns excluding the ID column
    # We'll first guess sample_cols as all columns except a likely ID col.
    # Determine sample cols by: columns with names starting with "GTEX-" (most reliable)
    gtex_like = [c for c in bcols if c.startswith("GTEX-")]

    if len(gtex_like) > 0:
        sample_cols = gtex_like
        id_col = pick_id_column(bcols, sample_cols)
    else:
        # If not GTEX-like, assume first non-numeric column is ID.
        # Use Arrow types from first_batch
        fields = {f.name: str(f.type) for f in first_batch.schema}
        non_numeric = [c for c,t in fields.items() if ("string" in t or "large_string" in t)]
        # Pick first string column as ID if any
        id_col = non_numeric[0] if non_numeric else None
        # Everything else is sample columns
        sample_cols = [c for c in bcols if c != id_col]

    if id_col is None or id_col not in bcols:
        raise RuntimeError(
            f"Could not detect ID column. Columns seen: {bcols[:50]} ..."
        )

    n_samples = len(sample_cols)
    print("\nDetected wide format:")
    print("  id_col   :", id_col)
    print("  samples  :", n_samples)
    print("  features :", n_features)

    # Save sample IDs
    with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as f:
        for s in sample_cols:
            f.write(s + "\n")

    # Create memmap: samples x features
    mmap_path = os.path.join(out_dir, "X_samples_x_features.float32.mmap")
    X = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=(n_samples, n_features))
    X[:] = 0
    X.flush()

    # Save feature IDs streamed
    feat_path = os.path.join(out_dir, "features.txt")
    feat_file = open(feat_path, "w", encoding="utf-8")

    row_offset = 0
    print("\nStreaming parquet -> memmap ...")

    for rg in range(pf.metadata.num_row_groups):
        it = pf.iter_batches(
            row_groups=[rg],
            batch_size=batch_rows,
            columns=[id_col] + sample_cols
        )

        for batch in tqdm(it, desc=f"row_group {rg}"):
            # Avoid pandas KeyError issues by working in Arrow first
            arr_id = batch.column(0)  # id_col is first
            # write feature ids
            for v in arr_id.to_pylist():
                feat_file.write(str(v) + "\n")

            # remaining columns are samples
            # Convert to numpy: shape (batch_rows x n_samples)
            M = np.column_stack([np.asarray(batch.column(i)) for i in range(1, batch.num_columns)]).astype(np.float32, copy=False)

            r0 = row_offset
            r1 = row_offset + M.shape[0]
            X[:, r0:r1] = M.T
            row_offset = r1
            X.flush()

    feat_file.close()

    if row_offset != n_features:
        raise RuntimeError(f"Wrote {row_offset} features but expected {n_features}.")

    meta = {
        "parquet_path": os.path.abspath(parquet_path),
        "shape": [int(n_samples), int(n_features)],
        "mmap_path": os.path.abspath(mmap_path),
        "format": "wide (samples are columns, first column is feature id)",
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
        print("Usage: python scripts/02_wideparquet_to_memmap.py <parquet_path> [out_dir]")
        raise SystemExit(2)
    parquet = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "out"
    main(parquet, out_dir=out_dir)