#!/usr/bin/env python3
"""Convert CADBUSI SQLite tables to Parquet format for fast loading.

This script creates a Parquet cache of the database tables, enabling
~100x faster data loading for export scripts.

Usage:
    # Create cache (skips existing files)
    python cache_parquet.py /path/to/cadbusi.db

    # Force rebuild all cached files
    python cache_parquet.py /path/to/cadbusi.db --rebuild

    # Check cache status only
    python cache_parquet.py /path/to/cadbusi.db --status
"""

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional

import polars as pl


# Tables to cache
TABLES = ["Images", "StudyCases", "Lesions", "Pathology", "Videos"]


def get_cache_dir(db_path: Path) -> Path:
    """Get cache directory for Parquet files."""
    return db_path.parent / f".{db_path.stem}_parquet_cache"


def get_db_tables(db_path: Path) -> set:
    """Get list of tables in the database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    return tables


def cache_table(
    db_path: Path,
    table: str,
    cache_dir: Path,
    existing_tables: set,
) -> Optional[Path]:
    """Cache a single table to Parquet. Returns path or None if unavailable."""
    parquet_path = cache_dir / f"{table.lower()}.parquet"

    # Check if table exists in database
    if table not in existing_tables:
        # Try to find a CSV file as fallback
        csv_path = db_path.parent / f"{table}Data.csv"
        if csv_path.exists():
            print(f"  {table}: Loading from CSV...", end=" ", flush=True)
            start = time.time()
            df = pl.read_csv(csv_path, infer_schema_length=10000)
            df.write_parquet(parquet_path, compression="snappy")
            print(f"{len(df):,} rows in {time.time() - start:.1f}s")
            return parquet_path
        else:
            print(f"  {table}: Not available (no table or CSV)")
            return None

    # Convert from SQLite
    print(f"  {table}: Converting from SQLite...", end=" ", flush=True)
    start = time.time()

    try:
        # Try connectorx (fastest)
        df = pl.read_database_uri(
            query=f"SELECT * FROM {table}",
            uri=f"sqlite://{db_path}"
        )
    except Exception:
        # Fall back to pandas
        import pandas as pd
        conn = sqlite3.connect(str(db_path))
        pdf = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        df = pl.from_pandas(pdf)

    df.write_parquet(parquet_path, compression="snappy")
    print(f"{len(df):,} rows in {time.time() - start:.1f}s")
    return parquet_path


def ensure_cache(db_path: Path, rebuild: bool = False) -> Dict[str, Optional[Path]]:
    """Ensure Parquet cache exists for all tables.

    Args:
        db_path: Path to the SQLite database
        rebuild: If True, rebuild all cache files

    Returns:
        Dict mapping table names to Parquet paths (None if unavailable)
    """
    cache_dir = get_cache_dir(db_path)
    cache_dir.mkdir(exist_ok=True)

    existing_tables = get_db_tables(db_path)
    paths = {}

    for table in TABLES:
        parquet_path = cache_dir / f"{table.lower()}.parquet"

        if parquet_path.exists() and not rebuild:
            print(f"  {table}: Using cached")
            paths[table] = parquet_path
            continue

        paths[table] = cache_table(db_path, table, cache_dir, existing_tables)

    return paths


def get_cache_status(db_path: Path) -> Dict[str, dict]:
    """Get status of cached files."""
    cache_dir = get_cache_dir(db_path)
    existing_tables = get_db_tables(db_path)
    status = {}

    for table in TABLES:
        parquet_path = cache_dir / f"{table.lower()}.parquet"
        csv_path = db_path.parent / f"{table}Data.csv"

        info = {
            "in_database": table in existing_tables,
            "csv_available": csv_path.exists(),
            "cached": parquet_path.exists(),
            "cache_path": str(parquet_path) if parquet_path.exists() else None,
        }

        if parquet_path.exists():
            stat = parquet_path.stat()
            info["cache_size_mb"] = stat.st_size / (1024 * 1024)
            info["cache_modified"] = time.ctime(stat.st_mtime)

        status[table] = info

    return status


def main():
    parser = argparse.ArgumentParser(
        description="Cache CADBUSI tables as Parquet files"
    )
    parser.add_argument("db_path", type=Path, help="Path to cadbusi.db")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild all cache files")
    parser.add_argument("--status", action="store_true", help="Show cache status only")
    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Database not found: {args.db_path}")
        return 1

    if args.status:
        print(f"Cache status for: {args.db_path}")
        print(f"Cache directory: {get_cache_dir(args.db_path)}")
        print()

        status = get_cache_status(args.db_path)
        for table, info in status.items():
            cached = "✓" if info["cached"] else "✗"
            source = "DB" if info["in_database"] else ("CSV" if info["csv_available"] else "N/A")
            size = f"{info.get('cache_size_mb', 0):.1f}MB" if info["cached"] else "-"
            print(f"  {cached} {table:12} src={source:3} size={size}")

        return 0

    print(f"Building Parquet cache for: {args.db_path}")
    print(f"Cache directory: {get_cache_dir(args.db_path)}")
    print()

    start = time.time()
    paths = ensure_cache(args.db_path, rebuild=args.rebuild)

    print()
    print(f"Cache complete in {time.time() - start:.1f}s")

    available = sum(1 for p in paths.values() if p is not None)
    print(f"  {available}/{len(TABLES)} tables cached")

    return 0


if __name__ == "__main__":
    exit(main())
