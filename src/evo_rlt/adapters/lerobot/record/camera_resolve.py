"""Resolve a camera's USB serial to the ``/dev/video*`` node carrying its color stream.

Recording addresses cameras by ``index_or_path``, and a bare index (``"port": 6``)
is the one form that can go wrong silently. ``/dev/videoN`` numbering follows kernel
enumeration order, so replugging in a different order renames every node; the
recorder still opens three cameras and still writes three video streams, but
``left_wrist`` and ``right_wrist`` have swapped. Nothing errors, and the mistake is
close to undiscoverable once it is baked into a dataset.

Serial numbers are stable, so nodes are found by walking each node's sysfs parents
up to the USB device that owns a ``serial`` attribute. ``/dev/v4l/by-id/`` would be
the obvious shortcut, but it is not reliable here: the Gemini 335's colour node was
observed with a by-id link one hour and without one the next, while its siblings
kept theirs and shifted which node they pointed at. sysfs has an entry for every
node, always.

The remaining problem is that one depth camera claims six to eight nodes -- depth,
left/right IR, colour -- and only one of them is the RGB stream a policy trains on.
Picking the wrong one yields a plausible-looking greyscale or noise video rather
than an error. So the node is chosen by asking V4L2 which pixel formats it offers
and keeping the one that advertises a colour format.

Resolution runs at every process start, so returning a bare ``/dev/videoN`` is safe
even though that number is not stable across replugs.

The format query is a raw ``VIDIOC_ENUM_FMT`` ioctl rather than a shell out to
``v4l2-ctl``, which is packaged separately (``v4l-utils``) and is not guaranteed to
be installed on a robot host.

Usage:
    python -m evo_rlt.adapters.lerobot.record.camera_resolve
"""

from __future__ import annotations

import fcntl
import argparse
import json
import logging
import time
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

V4L_CLASS_DIR = Path("/sys/class/video4linux")

# VIDIOC_ENUM_FMT = _IOWR('V', 2, struct v4l2_fmtdesc), sizeof(v4l2_fmtdesc) == 64
_VIDIOC_ENUM_FMT = (3 << 30) | (64 << 16) | (ord("V") << 8) | 2
_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1

# Formats a colour stream is delivered in. Depth (``Z16``), IR (``GREY``, ``Y8I``,
# ``Y12I``) and the Bayer/planar IR variants are deliberately absent: those nodes
# open and stream happily, they just are not the picture.
COLOR_FOURCCS = frozenset({"YUYV", "MJPG", "UYVY", "NV12", "YU12", "RGB3", "BGR3", "H264"})

# UYVY and NV12 also appear on IR nodes alongside a greyscale format. A node is only
# treated as colour if it offers a colour format and no greyscale one.
MONO_FOURCCS = frozenset({"GREY", "Y8I ", "Y12I", "Z16 ", "BA81"})


def _fourccs(dev: Path) -> list[str]:
    """Pixel formats advertised by a capture node, as 4-character codes."""
    formats: list[str] = []
    try:
        fd = dev.open("rb")
    except OSError as exc:
        logger.debug("cannot open %s: %s", dev, exc)
        return formats
    with fd:
        for index in range(32):
            buf = bytearray(64)
            struct.pack_into("II", buf, 0, index, _V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(fd, _VIDIOC_ENUM_FMT, buf)
            except OSError:
                break  # EINVAL marks the end of the list
            (pixelformat,) = struct.unpack_from("I", buf, 44)
            formats.append(pixelformat.to_bytes(4, "little").decode("ascii", "replace"))
    return formats


def _is_color_node(dev: Path) -> bool:
    fourccs = _fourccs(dev)
    if not fourccs:
        return False
    if any(f in MONO_FOURCCS for f in fourccs):
        return False
    return any(f in COLOR_FOURCCS for f in fourccs)


def _serial_of_node(node: Path) -> str | None:
    """USB serial owning a ``/sys/class/video4linux/videoN`` entry, if any."""
    path = (node / "device").resolve()
    while path != path.parent:
        if (path / "serial").is_file() and (path / "idVendor").is_file():
            try:
                return (path / "serial").read_text().strip()
            except OSError:
                return None
        path = path.parent
    return None


def nodes_by_serial() -> dict[str, list[Path]]:
    """``/dev/videoN`` capture nodes grouped by the serial of the camera owning them."""
    out: dict[str, list[Path]] = {}
    if not V4L_CLASS_DIR.is_dir():
        return out
    for entry in sorted(V4L_CLASS_DIR.iterdir(), key=lambda p: int(p.name.removeprefix("video"))):
        serial = _serial_of_node(entry)
        if serial:
            out.setdefault(serial, []).append(Path("/dev") / entry.name)
    return out


def color_node_for_serial(serial: str) -> Path:
    """Return the ``/dev/videoN`` carrying ``serial``'s colour stream."""
    groups = nodes_by_serial()
    candidates = groups.get(serial)
    if not candidates:
        raise ValueError(
            f"No camera with serial {serial!r}. Connected serials: {sorted(groups)}"
        )

    color = [n for n in candidates if _is_color_node(n)]
    if not color:
        raise ValueError(
            f"Camera {serial!r} has {len(candidates)} node(s) but none advertise a colour "
            f"format: {[(n.name, _fourccs(n)) for n in candidates]}"
        )
    if len(color) > 1:
        logger.warning(
            "Camera %s exposes %d colour nodes (%s); using the first.",
            serial,
            len(color),
            [n.name for n in color],
        )
    return color[0]


def resolve_camera_port(port: object) -> int | Path:
    """Normalise a manifest ``port`` into something ``OpenCVCameraConfig`` accepts.

    Accepts, in order of preference:

    * ``{"serial": "235123073792"}`` or a bare serial string -- resolved here;
    * an explicit ``/dev/...`` path -- passed through as a ``Path``;
    * an ``int`` -- passed through, but warned about, since indices are unstable.
    """
    if isinstance(port, dict):
        serial = port.get("serial")
        if not serial:
            raise ValueError(f"camera port object needs a 'serial' key, got {port!r}")
        return color_node_for_serial(str(serial))

    if isinstance(port, int):
        logger.warning(
            "Camera addressed by index %d. /dev/videoN numbering follows kernel "
            "enumeration order and changes on replug, which silently swaps cameras "
            "in recorded data. Use {\"serial\": \"...\"} instead.",
            port,
        )
        return port

    text = str(port)
    if text.startswith("/dev/"):
        return Path(text)
    if text.isdigit():
        logger.warning("Camera addressed by index %s; prefer a serial.", text)
        return int(text)
    return color_node_for_serial(text)


WARMUP_FRAMES = 12
DEFAULT_MANIFEST = Path("configs/crp_dual_manifest.json")


def aliases_by_serial(manifest: Path = DEFAULT_MANIFEST) -> dict[str, str]:
    """serial -> alias from a manifest, so output names cameras the way configs do."""
    try:
        cams = json.loads(manifest.read_text()).get("cameras", [])
    except (OSError, ValueError):
        return {}
    out = {}
    for cam in cams:
        port = cam.get("port")
        serial = port.get("serial") if isinstance(port, dict) else None
        if serial:
            out[str(serial)] = cam.get("alias", "?")
    return out


def _grab(dev: Path):
    """One frame from a device, via the same camera class recording uses.

    Not a bare ``cv2.VideoCapture``: opening a RealSense that way yields a uniformly
    green image (all-zero YUV) unless something has already initialised it, which is
    a picture rather than an error and looks exactly like a broken camera.
    ``OpenCVCamera`` applies the fourcc/size/fps handshake that makes the sensor
    actually start, so this sees what the recorder would see.

    Opened and closed one camera at a time -- these share a USB controller and
    concurrent opens are unreliable.
    """
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    fmts = _fourccs(dev)
    fourcc = next((f for f in ("YUYV", "MJPG") if f in fmts), None)
    cam = OpenCVCamera(
        OpenCVCameraConfig(index_or_path=dev, width=640, height=480, fps=30,
                           fourcc=fourcc, warmup_s=2)
    )
    try:
        cam.connect()
    except Exception as exc:
        print(f"    ({type(exc).__name__}: {str(exc)[:70]})")
        return None
    try:
        rgb = cam.async_read(timeout_ms=2000)
    except Exception:
        return None
    finally:
        try:
            cam.disconnect()
        except Exception:
            pass
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # OpenCVCamera yields RGB


def _snapshot(rows: list[tuple[str, str, Path]], out_dir: Path) -> int:
    """Save one frame per named camera, plus a labelled contact sheet."""
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for alias, serial, dev in rows:
        frame = _grab(dev)
        if frame is None:
            print(f"  {alias:<14} {dev.name:<10} no frame")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            cv2.imwrite(str(out_dir / f"{alias}.png"), frame)
            print(f"  {alias:<14} {dev.name:<10} -> {out_dir / (alias + '.png')}")
        tile = cv2.resize(frame, (640, 480))
        cv2.rectangle(tile, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(tile, f"{alias}  {dev.name}  {serial}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    if not tiles:
        print("  no named cameras in the manifest")
        return 1
    sheet = out_dir / "all.png"
    cv2.imwrite(str(sheet), np.hstack(tiles))
    print(f"\n  contact sheet: {sheet}")
    # The RealSense v4l2 nodes stay busy for a while after release; starting a
    # recording immediately hits "Timed out waiting for frame".
    print("  wait ~15s before starting a recording (v4l2 nodes release slowly)")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    ap = argparse.ArgumentParser(description="Map camera serials to /dev/video nodes.")
    ap.add_argument("--snapshot", action="store_true", help="save one frame per named camera")
    ap.add_argument("--out", type=Path, default=Path("outputs/camera_check"))
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    groups = nodes_by_serial()
    if not groups:
        print("No V4L2 capture devices found.")
        return 1
    names = aliases_by_serial(args.manifest)

    rows: list[tuple[str, str, Path]] = []
    print(f"{'name':<14} {'serial':<20} {'color node':<12} formats")
    for serial in sorted(groups, key=lambda s: (names.get(s, "zz"), s)):
        alias = names.get(serial, "-")
        try:
            dev = color_node_for_serial(serial)
        except ValueError as exc:
            print(f"{alias:<14} {serial:<20} {'-':<12} {exc}")
            continue
        print(f"{alias:<14} {serial:<20} {dev.name:<12} {','.join(_fourccs(dev))}")
        if alias != "-":
            rows.append((alias, serial, dev))

    return _snapshot(rows, args.out) if args.snapshot else 0


if __name__ == "__main__":
    raise SystemExit(main())
