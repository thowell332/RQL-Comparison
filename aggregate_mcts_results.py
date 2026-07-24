#!/usr/bin/env python3
"""Combine MCTS eval CSVs, drop duplicate episode rows, and print aggregates.

``eval_mcts.py`` appends one row per finished episode. Parallel workers (and
overlapping resume/shard runs) can write the same ``(seed, episode_id)`` more
than once. This script keeps one row per key and summarizes like ``eval_mcts``.

Examples:
  python aggregate_mcts_results.py path/to/episodes.csv
  python aggregate_mcts_results.py run_a/episodes.csv run_b/episodes.csv
  python aggregate_mcts_results.py eval_mcts/mcts_3l30v_seed42/ -o combined.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Match eval_mcts.CSV_FIELDS
CSV_FIELDS = [
    "episode_id",
    "seed",
    "success",
    "episode_length",
    "episode_reward",
    "basic_reward",
    "mean_speed",
    "mean_lane",
    "mean_courtesy_gap",
    "courtesy_active_steps",
    "total_norm_cost",
]

NUMERIC_COMPARE_FIELDS = [
    "success",
    "episode_length",
    "episode_reward",
    "basic_reward",
    "mean_speed",
    "mean_lane",
    "mean_courtesy_gap",
    "courtesy_active_steps",
    "total_norm_cost",
]


def _as_float(value: str) -> float:
    return float(value)


def _row_key(row: dict) -> tuple[int, int]:
    return int(float(row["seed"])), int(float(row["episode_id"]))


def _rows_conflict(a: dict, b: dict, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """True if two rows share a key but disagree on metrics."""
    for field in NUMERIC_COMPARE_FIELDS:
        if field not in a or field not in b:
            continue
        try:
            va, vb = float(a[field]), float(b[field])
        except (TypeError, ValueError):
            if str(a[field]) != str(b[field]):
                return True
            continue
        if not np.isclose(va, vb, rtol=rtol, atol=atol):
            return True
    return False


def collect_csv_paths(inputs: Iterable[Path]) -> list[Path]:
    """Expand files and directories into episode CSV paths."""
    paths: list[Path] = []
    seen: set[Path] = set()

    def _is_episode_csv(candidate: Path) -> bool:
        name = candidate.name.lower()
        if candidate.suffix.lower() != ".csv":
            return False
        if "combined" in name:
            return False
        return name == "episodes.csv" or name.startswith("episodes_")

    for raw in inputs:
        path = raw.expanduser().resolve()
        candidates: list[Path] = []
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            named = sorted(
                p for p in path.rglob("episodes*.csv") if _is_episode_csv(p)
            )
            candidates = named if named else [
                p for p in sorted(path.rglob("*.csv")) if _is_episode_csv(p)
            ]
        else:
            raise FileNotFoundError(f"Input not found: {path}")

        for candidate in candidates:
            if candidate in seen:
                continue
            # Allow explicitly passed files even if name is atypical.
            if path.is_dir() and not _is_episode_csv(candidate):
                continue
            seen.add(candidate)
            paths.append(candidate)
    return paths


def load_and_dedupe(
    csv_paths: list[Path],
    keep: str = "first",
) -> tuple[list[dict], dict]:
    """Load CSVs and keep one row per (seed, episode_id).

    keep: "first" | "last" — which duplicate to retain (file order, then row order).
    """
    if keep not in ("first", "last"):
        raise ValueError(f"keep must be 'first' or 'last', got {keep!r}")

    by_key: dict[tuple[int, int], dict] = {}
    source_counts: dict[str, int] = {}
    raw_rows = 0
    duplicate_rows = 0
    conflicting_duplicates = 0

    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            missing = [f for f in ("episode_id", "seed") if f not in reader.fieldnames]
            if missing:
                raise ValueError(f"{csv_path} missing required columns: {missing}")

            file_rows = 0
            for row in reader:
                raw_rows += 1
                file_rows += 1
                key = _row_key(row)
                cleaned = {field: row.get(field, "") for field in CSV_FIELDS}
                cleaned["_source"] = str(csv_path)

                if key not in by_key:
                    by_key[key] = cleaned
                    continue

                duplicate_rows += 1
                if _rows_conflict(by_key[key], cleaned):
                    conflicting_duplicates += 1
                if keep == "last":
                    by_key[key] = cleaned

            source_counts[str(csv_path)] = file_rows

    rows = [by_key[k] for k in sorted(by_key.keys())]
    stats = {
        "n_files": len(csv_paths),
        "raw_rows": raw_rows,
        "unique_rows": len(rows),
        "duplicate_rows": duplicate_rows,
        "conflicting_duplicates": conflicting_duplicates,
        "source_counts": source_counts,
    }
    return rows, stats


def write_combined_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def print_aggregate(rows: list[dict], stats: dict) -> None:
    print("________________________________________________")
    print(f"Files: {stats['n_files']}")
    for path, count in stats["source_counts"].items():
        print(f"  {path}: {count} row(s)")
    print(f"Raw rows: {stats['raw_rows']}")
    print(f"Duplicates dropped: {stats['duplicate_rows']}")
    if stats["conflicting_duplicates"]:
        print(
            f"WARNING: {stats['conflicting_duplicates']} duplicate key(s) "
            f"had conflicting metric values (kept per --keep)."
        )
    print(f"Unique episodes: {stats['unique_rows']}")

    if not rows:
        print("No completed episodes to summarize.")
        print("________________________________________________")
        return

    episode_ids = sorted(int(float(r["episode_id"])) for r in rows)
    seeds = sorted({int(float(r["seed"])) for r in rows})
    print(f"Episode id range: [{episode_ids[0]}, {episode_ids[-1]}]")
    if len(seeds) <= 8:
        print(f"Seeds: {seeds}")
    else:
        print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} distinct)")

    # Coverage gaps by episode_id (seed is typically base_seed + episode_id).
    id_set = set(episode_ids)
    missing = [i for i in range(episode_ids[0], episode_ids[-1] + 1) if i not in id_set]
    if missing:
        preview = missing[:20]
        suffix = " ..." if len(missing) > 20 else ""
        print(
            f"Missing episode_ids within "
            f"[{episode_ids[0]}, {episode_ids[-1]}]: {preview}{suffix} "
            f"({len(missing)} total)"
        )

    successes = np.array([_as_float(r["success"]) for r in rows], dtype=float)
    added = np.array([_as_float(r["episode_reward"]) for r in rows], dtype=float)
    basic = np.array([_as_float(r["basic_reward"]) for r in rows], dtype=float)
    lengths = np.array([_as_float(r["episode_length"]) for r in rows], dtype=float)
    speeds = np.array([_as_float(r["mean_speed"]) for r in rows], dtype=float)
    lanes = np.array([_as_float(r["mean_lane"]) for r in rows], dtype=float)
    costs = np.array([_as_float(r["total_norm_cost"]) for r in rows], dtype=float)

    print("-----------")
    print(f"Success rate: {100.0 * np.mean(successes):.2f}%")
    print(f"{len(rows)} Episodes")
    print(f"Mean added reward: {np.mean(added):.2f} +/- {np.std(added):.2f}")
    print(f"Mean basic reward: {np.mean(basic):.2f} +/- {np.std(basic):.2f}")
    print(f"Mean episode length: {np.mean(lengths):.2f} +/- {np.std(lengths):.2f}")
    print(f"Mean lane: {np.mean(lanes):.2f} +/- {np.std(lanes):.2f}")
    print(f"Mean speed: {np.mean(speeds):.2f} +/- {np.std(speeds):.2f}")
    print(f"Mean total norm cost: {np.mean(costs):.2f} +/- {np.std(costs):.2f}")
    if rows and "mean_courtesy_gap" in rows[0]:
        gaps = np.array(
            [
                _as_float(r["mean_courtesy_gap"])
                if r.get("mean_courtesy_gap") not in ("", None)
                else float("nan")
                for r in rows
            ],
            dtype=float,
        )
        if np.any(np.isfinite(gaps)):
            finite = gaps[np.isfinite(gaps)]
            print(
                f"Mean courtesy gap: {np.mean(finite):.2f} +/- {np.std(finite):.2f} "
                f"({len(finite)}/{len(gaps)} episodes with active gate)"
            )
    print("________________________________________________")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Combine MCTS episodes.csv files, drop duplicate "
        "(seed, episode_id) rows, and print aggregate metrics."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more episodes.csv files, or directories containing them.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the deduplicated combined CSV.",
    )
    parser.add_argument(
        "--keep",
        choices=("first", "last"),
        default="first",
        help="Which duplicate (seed, episode_id) row to keep (default: first).",
    )
    args = parser.parse_args(argv)

    try:
        csv_paths = collect_csv_paths(args.inputs)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not csv_paths:
        print("No CSV files found in the given inputs.", file=sys.stderr)
        return 1

    rows, stats = load_and_dedupe(csv_paths, keep=args.keep)
    print_aggregate(rows, stats)

    if args.output is not None:
        write_combined_csv(rows, args.output.expanduser().resolve())
        print(f"Wrote {len(rows)} unique row(s) to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
