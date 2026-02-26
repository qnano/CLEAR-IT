#!/usr/bin/env python3
"""Merge many Excel workbooks into one workbook with one tab per source sheet.

Default behavior targets the main plotting outputs in:
  notebooks/98_plotting/01_main/*.xlsx

Key features:
- Preserves all sheets from all source workbooks.
- Uses deterministic merged sheet names derived from source workbook + source sheet.
- Handles Excel constraints (invalid chars, 31-char limit, uniqueness).
- Adds README and manifest tabs for traceability.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

EXCEL_SHEET_MAX_LEN = 31
INVALID_SHEET_CHARS = re.compile(r"[\\/*?:\[\]]")


@dataclass
class SheetJob:
    source_file: Path
    source_sheet: str
    merged_sheet: str
    rows: int
    cols: int


def sanitize_sheet_name(name: str) -> str:
    clean = INVALID_SHEET_CHARS.sub("_", name).strip()
    return clean or "Sheet"


def unique_sheet_name(base: str, used: Set[str]) -> str:
    base = sanitize_sheet_name(base)
    if len(base) > EXCEL_SHEET_MAX_LEN:
        base = base[:EXCEL_SHEET_MAX_LEN]

    if base not in used:
        used.add(base)
        return base

    i = 1
    while True:
        suffix = f"_{i:02d}"
        prefix_len = EXCEL_SHEET_MAX_LEN - len(suffix)
        candidate = f"{base[:prefix_len]}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def discover_workbooks(input_dir: Path, pattern: str, output_file: Path) -> List[Path]:
    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    out_resolved = output_file.resolve()

    filtered = []
    for p in files:
        # Ignore temporary lock files created by Excel
        if p.name.startswith("~$"):
            continue
        if p.resolve() == out_resolved:
            continue
        filtered.append(p)
    return filtered


def build_jobs(files: Iterable[Path]) -> List[SheetJob]:
    jobs: List[SheetJob] = []
    used_sheet_names: Set[str] = set()

    for file_path in files:
        excel = pd.ExcelFile(file_path)
        for sheet_name in excel.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            merged_base = f"{file_path.stem}__{sheet_name}"
            merged_name = unique_sheet_name(merged_base, used_sheet_names)
            jobs.append(
                SheetJob(
                    source_file=file_path,
                    source_sheet=sheet_name,
                    merged_sheet=merged_name,
                    rows=len(df),
                    cols=len(df.columns),
                )
            )

    return jobs


def write_merged_workbook(output_file: Path, jobs: List[SheetJob]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    readme = pd.DataFrame(
        {
            "field": [
                "description",
                "generated_utc",
                "sheet_naming",
                "note",
            ],
            "value": [
                "Consolidated source data workbook for plotting outputs.",
                ts,
                "<workbook_stem>__<source_sheet> (sanitized, unique, max 31 chars)",
                "See manifest tab for mapping from source workbook/sheet to merged sheet.",
            ],
        }
    )

    manifest = pd.DataFrame(
        [
            {
                "source_file": str(job.source_file),
                "source_sheet": job.source_sheet,
                "merged_sheet": job.merged_sheet,
                "rows": job.rows,
                "columns": job.cols,
            }
            for job in jobs
        ]
    )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        manifest.to_excel(writer, sheet_name="manifest", index=False)

        for job in jobs:
            df = pd.read_excel(job.source_file, sheet_name=job.source_sheet)
            df.to_excel(writer, sheet_name=job.merged_sheet, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple XLSX workbooks into one workbook (one tab per source sheet)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("notebooks/98_plotting/01_main"),
        help="Directory containing source XLSX files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.xlsx",
        help="Glob pattern for source files (default: *.xlsx).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/98_plotting/01_main/supplementary_data_main_figures.xlsx"),
        help="Path to consolidated output workbook.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned merge actions without writing output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    files = discover_workbooks(args.input_dir, args.pattern, args.output)
    if not files:
        print("No XLSX files found.")
        return 1

    jobs = build_jobs(files)

    print(f"Input workbooks: {len(files)}")
    print(f"Source sheets: {len(jobs)}")
    print(f"Output: {args.output}")

    if args.dry_run:
        for job in jobs:
            print(
                f"- {job.source_file.name}:{job.source_sheet} -> {job.merged_sheet} "
                f"({job.rows}x{job.cols})"
            )
        return 0

    write_merged_workbook(args.output, jobs)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
