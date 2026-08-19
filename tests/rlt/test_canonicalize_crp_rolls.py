from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evo_rlt.adapters.crp.action_angles import canonicalize_crp_roll_deg
from evo_rlt.cli.canonicalize_crp_rolls import canonicalize_dataset


def test_canonicalize_crp_roll_only_snaps_branch_cut() -> None:
    assert canonicalize_crp_roll_deg(180.0) == -180.0
    assert canonicalize_crp_roll_deg(179.998) == -180.0
    assert canonicalize_crp_roll_deg(-179.98) == -180.0
    assert canonicalize_crp_roll_deg(90.0) == 90.0


def test_dataset_copy_is_clean_and_source_is_untouched(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "clean"
    (source / "meta").mkdir(parents=True)
    (source / "data" / "chunk-000").mkdir(parents=True)
    (source / "videos").mkdir()
    names = ["left_x.pos", "left_roll.pos", "right_roll.pos"]
    info = {
        "features": {
            "action": {"names": names, "dtype": "float32", "shape": [3]},
        }
    }
    (source / "meta" / "info.json").write_text(json.dumps(info))
    stats = {
        "action": {
            "min": [1.0, -180.0, -180.0],
            "max": [2.0, 180.0, 180.0],
            "mean": [1.5, 0.0, 0.0],
            "std": [0.5, 180.0, 180.0],
            "count": [2.0],
        }
    }
    (source / "meta" / "stats.json").write_text(json.dumps(stats))
    source_actions = [[1.0, 180.0, -179.99], [2.0, -180.0, 179.998]]
    table = pa.table(
        {
            "action": pa.array(source_actions, type=pa.list_(pa.float32())),
            "episode_index": pa.array([0, 0], type=pa.int64()),
        }
    )
    source_parquet = source / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(table, source_parquet)
    (source / "videos" / "dummy.mp4").write_bytes(b"video")

    changed = canonicalize_dataset(source, output)

    assert changed["action"] == 3
    original = pq.read_table(source_parquet)["action"].to_pylist()
    cleaned = pq.read_table(output / "data" / "chunk-000" / "file-000.parquet")["action"].to_pylist()
    np.testing.assert_allclose(original, source_actions)
    np.testing.assert_allclose(cleaned, [[1.0, -180.0, -180.0], [2.0, -180.0, -180.0]])
    cleaned_stats = json.loads((output / "meta" / "stats.json").read_text())["action"]
    assert cleaned_stats["mean"] == [1.5, -180.0, -180.0]
    assert cleaned_stats["std"] == [0.5, 0.0, 0.0]

    # CLI mode edits an explicitly selected merged dataset in place and is
    # idempotent once the branch-cut values have been canonicalized.
    assert canonicalize_dataset(output)["action"] == 0
