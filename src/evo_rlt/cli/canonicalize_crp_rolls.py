"""Canonicalize CRP roll targets in an ACT training dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evo_rlt.adapters.crp.action_angles import canonicalize_crp_roll_deg


ACTION_COLUMNS = ("action", "complementary_info.policy_action")


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link large immutable videos, falling back to a regular copy."""

    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _roll_indices(info: dict, column: str) -> list[int]:
    feature = info.get("features", {}).get(column)
    if feature is None:
        return []
    names = feature.get("names") or []
    return [index for index, name in enumerate(names) if name.endswith("_roll.pos")]


def _canonicalize_array(values: np.ndarray, roll_indices: list[int]) -> tuple[np.ndarray, int]:
    output = np.asarray(values, dtype=np.float32).copy()
    changed = 0
    for index in roll_indices:
        before = output[:, index].copy()
        output[:, index] = np.fromiter(
            (canonicalize_crp_roll_deg(value) for value in before),
            dtype=np.float32,
            count=len(before),
        )
        changed += int(np.count_nonzero(output[:, index] != before))
    return output, changed


def _replace_list_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    field = table.schema.field(name)
    replacement = pa.array(values.tolist(), type=field.type)
    return table.set_column(table.schema.get_field_index(name), field, replacement)


def _feature_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [float(values.shape[0])],
    }


def _rewrite_episode_action_stats(
    output_root: Path,
    action_values: np.ndarray,
    episode_indices: np.ndarray,
) -> None:
    episode_actions = {
        int(episode): action_values[episode_indices == episode]
        for episode in np.unique(episode_indices)
    }
    stat_functions = {
        "min": lambda values: values.min(axis=0),
        "max": lambda values: values.max(axis=0),
        "mean": lambda values: values.mean(axis=0, dtype=np.float64),
        "std": lambda values: values.std(axis=0, dtype=np.float64),
        "q01": lambda values: np.quantile(values, 0.01, axis=0),
        "q10": lambda values: np.quantile(values, 0.10, axis=0),
        "q50": lambda values: np.quantile(values, 0.50, axis=0),
        "q90": lambda values: np.quantile(values, 0.90, axis=0),
        "q99": lambda values: np.quantile(values, 0.99, axis=0),
    }
    for parquet_path in sorted((output_root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        episodes = np.asarray(table["episode_index"], dtype=np.int64)
        for stat_name, stat_function in stat_functions.items():
            column = f"stats/action/{stat_name}"
            if column not in table.column_names:
                continue
            values = np.stack([stat_function(episode_actions[int(ep)]) for ep in episodes])
            table = _replace_list_column(table, column, values)
        count_column = "stats/action/count"
        if count_column in table.column_names:
            counts = np.asarray(
                [[len(episode_actions[int(ep)])] for ep in episodes], dtype=np.int64
            )
            table = _replace_list_column(table, count_column, counts)
        temporary = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, parquet_path)


def canonicalize_dataset(
    source_root: Path,
    output_root: Path | None = None,
) -> dict[str, int]:
    source_root = source_root.expanduser().resolve()
    info_path = source_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset (missing {info_path})")

    info = json.loads(info_path.read_text())
    indices = {column: _roll_indices(info, column) for column in ACTION_COLUMNS}
    if not indices["action"]:
        raise ValueError("Dataset action feature has no *_roll.pos dimensions")

    if output_root is None:
        working_root = source_root
    else:
        working_root = output_root.expanduser().resolve()
        if working_root.exists():
            raise FileExistsError(f"Output already exists: {working_root}")
        if (
            working_root == source_root
            or working_root in source_root.parents
            or source_root in working_root.parents
        ):
            raise ValueError("Output must be separate from the source dataset")
        # Optional programmatic copy mode is retained for callers that want it.
        # Hard links avoid duplicating immutable videos; mutable parquet files are
        # atomically replaced below before any edits can reach the source inode.
        shutil.copytree(source_root, working_root, copy_function=_link_or_copy)

    totals = {column: 0 for column in ACTION_COLUMNS}
    collected: dict[str, list[np.ndarray]] = {column: [] for column in ACTION_COLUMNS}
    episode_indices: list[np.ndarray] = []
    for parquet_path in sorted((working_root / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        episode_indices.append(np.asarray(table["episode_index"], dtype=np.int64))
        for column in ACTION_COLUMNS:
            if column not in table.column_names or not indices[column]:
                continue
            values = np.asarray(table[column].combine_chunks().to_pylist(), dtype=np.float32)
            canonical, changed = _canonicalize_array(values, indices[column])
            table = _replace_list_column(table, column, canonical)
            totals[column] += changed
            collected[column].append(canonical)

        temporary = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, parquet_path)

    stats_path = working_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for column, chunks in collected.items():
        if chunks and column in stats:
            stats[column] = _feature_stats(np.concatenate(chunks, axis=0))
    temporary_stats = stats_path.with_suffix(".json.tmp")
    temporary_stats.write_text(json.dumps(stats, indent=2) + "\n")
    os.replace(temporary_stats, stats_path)
    if collected["action"] and episode_indices:
        _rewrite_episode_action_stats(
            working_root,
            np.concatenate(collected["action"], axis=0),
            np.concatenate(episode_indices),
        )
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "In-place canonicalization of CRP roll values at +/-180 degrees to "
            "exactly -180 degrees. Point --root at a merged training dataset, not "
            "at the original recording-session directories."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    totals = canonicalize_dataset(args.root)
    print(f"Canonicalized dataset in place: {args.root}")
    for column, changed in totals.items():
        print(f"  {column}: canonicalized {changed} roll values")


if __name__ == "__main__":
    main()
