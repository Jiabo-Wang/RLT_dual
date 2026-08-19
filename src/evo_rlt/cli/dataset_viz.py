"""Visualize a range of episodes in a single Rerun session.

``lerobot-dataset-viz`` takes exactly one ``--episode-index``, so reviewing a
recording session means relaunching the viewer once per episode. This entry
point takes an episode *range* instead and logs every selected episode into one
recording on a continuous timeline, so a session plays back end to end.

Episode boundaries stay visible: ``episode_index`` is logged as a scalar (a
staircase in the plot panel) and each episode start emits a text log entry.

A blueprint ships with the recording so all cameras are visible at once and each
feature gets its own plot; see :func:`build_blueprint`.

Examples:

```
# whole session, every 4th frame (fast skim)
evo-rlt-dataset-viz --root data/crp_dual/0817_screw_demo_v1/record_teleop_full_172919 --stride 4

# episodes 10..19 at full rate
evo-rlt-dataset-viz --root data/... --episodes 10-19

# plots only, no video decoding (cheap, near-instant)
evo-rlt-dataset-viz --root data/... --no-images
```
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.utils.data
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging

# Logged as timelines or episode labels rather than as plot series.
_SKIP_SCALARS = {"index", "task_index", "frame_index", "timestamp", "episode_index"}


def parse_episode_spec(spec: str, total: int) -> list[int]:
    """Expand ``"all"`` / ``"0-9,20,30-"`` into a sorted list of episode indices.

    ``-`` is always a range separator, never a sign: either end may be omitted,
    so ``"-4"`` is the first five episodes and ``"30-"`` runs to the end.
    """
    if spec.strip().lower() in {"all", "*", ""}:
        return list(range(total))

    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, _, end_s = part.partition("-")
            start = int(start_s) if start_s.strip() else 0
            end = int(end_s) if end_s.strip() else total - 1
        else:
            start = end = int(part)
        if start > end:
            raise ValueError(f"empty episode range {part!r}")
        selected.update(range(start, end + 1))

    out_of_range = sorted(i for i in selected if not 0 <= i < total)
    if out_of_range:
        shown = out_of_range if len(out_of_range) <= 4 else [out_of_range[0], "...", out_of_range[-1]]
        raise ValueError(f"episode {shown} outside 0..{total - 1}")
    return sorted(selected)


def build_blueprint(features: dict, camera_keys: list[str]) -> rrb.Blueprint:
    """Cameras side by side on top, plots below.

    Rerun's auto-layout drops the camera views into a tab container, so only one
    of them is visible at a time; sending a blueprint is the only way to get all
    three at once. Per-feature plots also beat the single auto plot, which piles
    all 40-odd series onto one axis.
    """
    scalars = [
        (key, ft)
        for key, ft in features.items()
        if key not in _SKIP_SCALARS and ft["dtype"] not in ("video", "image")
    ]
    # Vectors (action, states) each get their own axis; the 1-wide flags share one
    # with episode_index, since they are all read as staircases against the timeline.
    plots: list[rrb.View] = [
        rrb.TimeSeriesView(origin=key, name=key)
        for key, ft in scalars
        if int(np.prod(ft["shape"])) > 1
    ]
    flags = [f"{key}/**" for key, ft in scalars if int(np.prod(ft["shape"])) == 1]
    plots.append(rrb.TimeSeriesView(origin="/", contents=[*flags, "episode_index"], name="flags"))
    plots.append(rrb.TextLogView(origin="episode", name="episodes"))
    grid = rrb.Grid(contents=plots, grid_columns=2)

    if not camera_keys:
        return rrb.Blueprint(grid, collapse_panels=True)
    cameras = rrb.Horizontal(
        contents=[rrb.Spatial2DView(origin=key, name=key.split(".")[-1]) for key in camera_keys]
    )
    return rrb.Blueprint(rrb.Vertical(cameras, grid, row_shares=[1, 1]), collapse_panels=True)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    return (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()


def _scalar_keys(batch: dict) -> list[str]:
    keys = []
    for key, value in batch.items():
        if key in _SKIP_SCALARS or key.startswith("observation.images"):
            continue
        if isinstance(value, torch.Tensor) and value.ndim <= 2 and value.dtype != torch.bool:
            keys.append(key)
    return keys


def visualize(
    dataset: LeRobotDataset,
    episodes: list[int],
    *,
    stride: int = 1,
    batch_size: int = 32,
    num_workers: int = 4,
    show_images: bool = True,
    jpeg_quality: int = 85,
) -> None:
    indices = range(0, len(dataset), stride)
    view = torch.utils.data.Subset(dataset, list(indices))
    dataloader = torch.utils.data.DataLoader(
        view, num_workers=num_workers, batch_size=batch_size, shuffle=False
    )

    camera_keys = list(dataset.meta.camera_keys) if show_images else []
    fps = dataset.meta.fps
    scalar_keys: list[str] | None = None
    current_episode: int | None = None
    frame_counter = 0

    logging.info("Logging to Rerun")
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if scalar_keys is None:
            scalar_keys = _scalar_keys(batch)

        for i in range(len(batch["index"])):
            episode = int(batch["episode_index"][i].item())
            # One continuous timeline across the whole selection: playing it back
            # runs the episodes end to end at (stride-adjusted) real time.
            rr.set_time("frame", sequence=frame_counter)
            rr.set_time("timestamp", duration=frame_counter * stride / fps)
            rr.set_time("episode_frame", sequence=int(batch["frame_index"][i].item()))
            frame_counter += 1

            if episode != current_episode:
                current_episode = episode
                position = episodes.index(episode) + 1
                task = batch["task"][i] if "task" in batch else ""
                label = f"▶ episode {episode}  ({position}/{len(episodes)})"
                rr.log("episode", rr.TextLog(f"{label}  {task[:80]}".rstrip()))

            for key in camera_keys:
                img = rr.Image(to_hwc_uint8_numpy(batch[key][i]))
                rr.log(key, img.compress(jpeg_quality=jpeg_quality) if jpeg_quality else img)

            rr.log("episode_index", rr.Scalars(float(episode)))
            for key in scalar_keys:
                value = batch[key][i]
                if value.ndim == 0:
                    rr.log(key, rr.Scalars(value.item()))
                else:
                    for dim_idx, val in enumerate(value):
                        rr.log(f"{key}/{dim_idx}", rr.Scalars(val.item()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play a range of LeRobot episodes back to back in one Rerun viewer."
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Local dataset directory (the one holding meta/)."
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Defaults to local/<root directory name>, which is how evo-rlt-record names them.",
    )
    parser.add_argument(
        "--episodes",
        default="all",
        help=(
            "Episode selection: 'all', '15', '0-9', '0-4,10,20-' (default: all). "
            "Either end of a range may be omitted; a leading dash needs --episodes=-4."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Log every Nth frame. Skimming a long session is much cheaper at 4-8.",
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Skip video decoding; log states/actions only."
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="Re-encode frames before sending to the viewer. 0 sends them uncompressed.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--memory-limit",
        default="75%",
        help="Viewer RAM budget; it drops the oldest frames past this instead of dying.",
    )
    parser.add_argument(
        "--save", type=Path, default=None, help="Write a .rrd file instead of spawning a viewer."
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve over gRPC for a remote viewer instead of spawning one locally.",
    )
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    args = parser.parse_args()

    init_logging()
    # A wrong --root otherwise surfaces as a HuggingFace Hub 404 that says nothing
    # about the local path.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    root = args.root.expanduser()
    if not (root / "meta" / "info.json").exists():
        parser.error(f"{root} does not look like a LeRobot dataset (no meta/info.json)")
    repo_id = args.repo_id or f"local/{root.resolve().name}"

    meta = LeRobotDataset(repo_id, root=root, episodes=[0], tolerance_s=args.tolerance_s).meta
    try:
        episodes = parse_episode_spec(args.episodes, meta.total_episodes)
    except ValueError as err:
        parser.error(f"--episodes {args.episodes!r}: {err}")
    if not episodes:
        parser.error(f"--episodes {args.episodes!r} selected nothing")

    logging.info("Loading %d/%d episodes from %s", len(episodes), meta.total_episodes, root)
    dataset = LeRobotDataset(repo_id, root=root, episodes=episodes, tolerance_s=args.tolerance_s)

    total = len(dataset)
    logged = (total + args.stride - 1) // args.stride
    show_images = not args.no_images
    logging.info(
        "%d frames (%.1f min) -> logging %d at stride %d%s",
        total,
        total / meta.fps / 60,
        logged,
        args.stride,
        "" if show_images else ", images off",
    )
    if show_images and logged * len(meta.camera_keys) > 30_000:
        logging.warning(
            "%d images to log. Rerun keeps them in RAM -- consider --stride 4 or --no-images.",
            logged * len(meta.camera_keys),
        )
    blueprint = build_blueprint(
        dict(dataset.meta.info["features"]), list(meta.camera_keys) if show_images else []
    )
    if not show_images:
        # Video decoding is gated on meta.video_keys, so dropping the video features
        # makes --no-images skip the MP4s entirely instead of decoding then discarding.
        features = dataset.meta.info["features"]
        for key in [k for k, ft in features.items() if ft["dtype"] in ("video", "image")]:
            del features[key]

    rr.init(f"{repo_id}/episodes_{args.episodes}", spawn=False)
    gc.collect()  # rr.init + dataloader workers can otherwise hang on a blocking flush
    if args.save is None and not args.serve:
        rr.spawn(memory_limit=args.memory_limit)
    elif args.serve:
        server_uri = rr.serve_grpc(grpc_port=args.grpc_port, server_memory_limit=args.memory_limit)
        rr.serve_web_viewer(open_browser=False, web_port=args.web_port, connect_to=server_uri)
        logging.info("Connect with: rerun rerun+http://<IP>:%d/proxy", args.grpc_port)
    rr.send_blueprint(blueprint)

    visualize(
        dataset,
        episodes,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        show_images=show_images,
        jpeg_quality=args.jpeg_quality,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.save)
        logging.info("Wrote %s -- open it with: rerun %s", args.save, args.save)
    elif args.serve:
        logging.info("Serving. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
