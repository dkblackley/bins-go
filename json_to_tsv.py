#!/usr/bin/env python3
"""
json_to_tsv.py

Convert a JSON dict of {QID: [DOCID, DOCID, ...]} into a TSV:
QID<TAB>DOCID<TAB>RANK

Example:
  python3 json_to_tsv.py out.json bm25tree_formatted.tsv

Notes:
- Accepts QIDs as strings or ints; writes them as strings.
- Accepts DOCIDs as strings or ints; writes them as strings.
- Validates that each value is a list/array.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, List[Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"ERROR: Failed to parse JSON: {path}\n{e}") from e
    except OSError as e:
        raise SystemExit(f"ERROR: Cannot read file: {path}\n{e}") from e

    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: Expected top-level JSON object/dict, got {type(data).__name__}")

    # Normalize keys to strings and validate values are lists
    out: Dict[str, List[Any]] = {}
    for qid, docids in data.items():
        qid_str = str(qid)
        if not isinstance(docids, list):
            raise SystemExit(
                f"ERROR: For QID={qid_str}, expected a JSON array/list of DOCIDs, got {type(docids).__name__}"
            )
        out[qid_str] = docids
    return out


def write_tsv(mapping: Dict[str, List[Any]], out_path: Path) -> None:
    qids = list(mapping.keys())

    try:
        with out_path.open("w", encoding="utf-8", newline="") as out:
            for qid in qids:
                docids = mapping[qid]
                for rank, docid in enumerate(docids, start=1):
                    out.write(f"{qid}\t{docid}\t{rank}\n")
    except OSError as e:
        raise SystemExit(f"ERROR: Cannot write output file: {out_path}\n{e}") from e


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Convert {QID: [DOCID...]} JSON to QID DOCID RANK TSV.")
    p.add_argument("input_json", type=Path, help="Path to input JSON (e.g., out.json)")
    p.add_argument("output_tsv", type=Path, help="Path to output TSV (e.g., bm25tree_formatted.tsv)")

    args = p.parse_args(argv)

    mapping = load_json(args.input_json)
    write_tsv(mapping, args.output_tsv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
