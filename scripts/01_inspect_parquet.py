import sys
import pyarrow.parquet as pq

def main(path: str):
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    print("=== Parquet schema ===")
    for f in schema:
        print(f"{f.name}: {f.type}")

    # show first row group stats
    print("\n=== Parquet metadata ===")
    md = pf.metadata
    print("num_row_groups:", md.num_row_groups)
    print("num_rows:", md.num_rows)
    print("created_by:", md.created_by)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/01_inspect_parquet.py <path.parquet>")
        raise SystemExit(2)
    main(sys.argv[1])