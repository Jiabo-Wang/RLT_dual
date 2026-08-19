"""Merge local LeRobot datasets without concatenating MP4 files.

LeRobot 0.5.1 remuxes adjacent source videos into one MP4 during a merge. Some
H.264 files have duplicate DTS values at the boundary, causing PyAV to fail
with "non monotonically increasing dts". Setting a deliberately small output
video file limit makes the upstream aggregator rotate files at every source
video boundary, so videos are copied losslessly instead of concatenated.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge local LeRobot datasets while keeping source MP4 files separate."
    )
    parser.add_argument(
        "--input-parent",
        type=Path,
        required=True,
        action="append",
        dest="input_parents",
        metavar="DIR",
        help=(
            "Directory containing record_* LeRobot dataset directories. Repeat the flag "
            "to merge several parents (e.g. one recording day each), in the order given."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument(
        "--repo-id-prefix",
        default="local/session_",
        help="Prefix used to generate a unique local repo ID for each input.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the exact output directory first if it already exists.",
    )
    return parser.parse_args()


def discover_datasets(input_parents: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for input_parent in input_parents:
        found = sorted(
            path
            for path in input_parent.iterdir()
            if path.is_dir() and (path / "meta" / "info.json").is_file()
        )
        if not found:
            raise FileNotFoundError(f"No LeRobot datasets found under {input_parent}")
        roots.extend(found)
    resolved = [root.resolve() for root in roots]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same source dataset was discovered twice; check --input-parent values")
    return roots


def prepare_output(output_root: Path, input_parents: list[Path], overwrite: bool) -> None:
    output = output_root.resolve()
    for input_parent in input_parents:
        inputs = input_parent.resolve()
        if output == inputs or output in inputs.parents:
            raise ValueError("Output must not be an input directory or one of its parents")
    if not output.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. Remove it or rerun with --overwrite."
        )
    shutil.rmtree(output)


def main() -> None:
    args = parse_args()
    roots = discover_datasets(args.input_parents)
    prepare_output(args.output_root, args.input_parents, args.overwrite)

    # With several parents the session timestamps can repeat across days, so keep the
    # parent directory name in the generated repo ID to keep the log readable.
    keep_parent_name = len(args.input_parents) > 1
    repo_ids = []
    for root in roots:
        session = root.name.removeprefix("record_teleop_full_")
        repo_ids.append(
            f"{args.repo_id_prefix}{root.parent.name}_{session}"
            if keep_parent_name
            else f"{args.repo_id_prefix}{session}"
        )
    source_meta = [
        LeRobotDatasetMetadata(repo_id=repo_id, root=root)
        for repo_id, root in zip(repo_ids, roots, strict=True)
    ]
    expected_episodes = sum(meta.total_episodes for meta in source_meta)
    expected_frames = sum(meta.total_frames for meta in source_meta)

    print(f"Merging {len(roots)} datasets into {args.output_root}")
    for repo_id, root in zip(repo_ids, roots, strict=True):
        print(f"  {repo_id}: {root}")

    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=args.output_repo_id,
        roots=roots,
        aggr_root=args.output_root,
        # Every source video is larger than 1 MiB in this dataset. The low
        # threshold forces file rotation and avoids the broken remux path.
        video_files_size_in_mb=1,
    )

    merged = LeRobotDatasetMetadata(repo_id=args.output_repo_id, root=args.output_root)
    if merged.total_episodes != expected_episodes or merged.total_frames != expected_frames:
        raise RuntimeError(
            "Merged totals do not match inputs: "
            f"expected {expected_episodes} episodes/{expected_frames} frames, "
            f"got {merged.total_episodes} episodes/{merged.total_frames} frames"
        )
    print(
        f"Merge complete: {merged.total_episodes} episodes, "
        f"{merged.total_frames} frames, output={args.output_root}"
    )


if __name__ == "__main__":
    main()
