from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def _write_chunk(path: Path, header: Iterable[str], rows: Iterable[Iterable[str]]) -> None:
    """Write a single CSV chunk."""
    with path.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.writer(dest)
        writer.writerow(header)
        writer.writerows(rows)


def split_csv(input_file: str, output_dir: str, rows_per_file: int) -> None:
    """Split a CSV file into multiple smaller files.

    Parameters
    ----------
    input_file: str
        Path to the source CSV file.
    output_dir: str
        Directory where the split files will be written. Created if missing.
    rows_per_file: int
        Maximum number of data rows per split file.
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="", encoding="utf-8") as src:
        reader = csv.reader(src)
        header = next(reader)
        rows: list[list[str]] = []
        part = 1
        for idx, row in enumerate(reader, start=1):
            rows.append(row)
            if idx % rows_per_file == 0:
                _write_chunk(output_path / f"{input_path.stem}_part{part}.csv", header, rows)
                rows = []
                part += 1
        if rows:
            _write_chunk(output_path / f"{input_path.stem}_part{part}.csv", header, rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split CSV into smaller chunks")
    parser.add_argument("input_file", help="Path to the source CSV")
    parser.add_argument(
        "output_dir", help="Directory where the split CSV files will be saved"
    )
    parser.add_argument(
        "--rows", type=int, default=1000, help="Maximum data rows per file (default: 1000)"
    )

    args = parser.parse_args()
    split_csv(args.input_file, args.output_dir, args.rows)
