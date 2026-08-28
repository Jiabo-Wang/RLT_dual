"""The local metrics sink and the live viewer that reads it.

wandb is the only place the online trainer reported loss, and the deployment box
runs offline and is not logged in -- so `--wandb` gets dropped there and the
curves vanish. The JSONL sink makes them available with no login, and survives a
kill mid-episode because every row is flushed.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The viewer needs matplotlib; the JSONL sink itself does not. Skip the whole
# module rather than fail in an env that only runs training.
pytest.importorskip("matplotlib")

from diagnostics.watch_online_loss import _series, read_rows, summarize  # noqa: E402


class _FakeTrainer:
    """Only the two methods under test, so this needs no robot or policy."""

    from evo_rlt.adapters.lerobot.record.online_trainer import OnlineRLTrainer

    _log_jsonl = OnlineRLTrainer._log_jsonl

    def __init__(self, path):
        self._metrics_path = path


class TestJsonlSink:
    def test_rows_are_appended_one_per_call(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        t = _FakeTrainer(path)
        t._log_jsonl({"online_rl/loss_critic": 1.5}, step=1)
        t._log_jsonl({"online_rl/loss_critic": 1.2}, step=2)
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert [r["step"] for r in rows] == [1, 2]
        assert rows[1]["online_rl/loss_critic"] == 1.2
        assert all("wall_time" in r for r in rows)

    def test_tensor_scalars_are_unwrapped(self, tmp_path):
        torch = pytest.importorskip("torch")
        path = tmp_path / "m.jsonl"
        _FakeTrainer(path)._log_jsonl({"loss": torch.tensor(0.25)}, step=0)
        assert json.loads(path.read_text())["loss"] == pytest.approx(0.25)

    def test_unserializable_values_are_dropped_not_fatal(self, tmp_path):
        path = tmp_path / "m.jsonl"
        _FakeTrainer(path)._log_jsonl({"ok": 1.0, "bad": object()}, step=0)
        row = json.loads(path.read_text())
        assert row["ok"] == 1.0 and "bad" not in row

    def test_an_unwritable_path_disables_the_sink_instead_of_raising(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        t = _FakeTrainer(blocker / "nested" / "m.jsonl")
        t._log_jsonl({"x": 1.0}, step=0)  # must not raise
        assert t._metrics_path is None  # and must not keep retrying every episode

    def test_no_sink_configured_is_a_noop(self):
        _FakeTrainer(None)._log_jsonl({"x": 1.0}, step=0)


class TestViewerReads:
    def test_a_half_written_final_line_is_skipped(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text('{"step": 1, "a": 1.0}\n{"step": 2, "a": 2.0}\n{"step": 3, "a')
        rows = read_rows(path)
        assert [r["step"] for r in rows] == [1, 2]

    def test_series_skips_rows_missing_the_metric(self, tmp_path):
        rows = [{"step": 1, "a": 1.0}, {"step": 2}, {"step": 3, "a": 3.0}]
        steps, values = _series(rows, "a")
        assert list(steps) == [1, 3] and list(values) == [1.0, 3.0]

    def test_series_of_an_absent_metric_is_empty(self):
        steps, values = _series([{"step": 1}], "nope")
        assert steps.size == 0 and values.size == 0

    def test_summary_reports_dashes_before_training_starts(self):
        # Warmup logs buffer growth but no loss; the summary must still render.
        line = summarize([{"step": 3, "online_rl/buffer_transitions": 400}])
        assert "ep    3" in line and "—" in line

    def test_summary_of_an_empty_run(self):
        assert summarize([]) == "还没有指标"


class TestRendering:
    def _render(self, tmp_path, rows):
        from diagnostics.watch_online_loss import render

        out = tmp_path / "live.png"
        render(rows, out)
        assert out.is_file() and out.stat().st_size > 0
        # The atomic write must not leave its temporary behind.
        assert not (tmp_path / "live.partial.png").exists()
        return out

    def test_warmup_only_run_renders(self, tmp_path):
        self._render(tmp_path, [
            {"step": i, "online_rl/buffer_transitions": i * 100,
             "online_rl/warmup_satisfied": 0} for i in range(1, 5)
        ])

    def test_full_run_renders(self, tmp_path):
        self._render(tmp_path, [
            {"step": i, "online_rl/buffer_transitions": i * 200,
             "online_rl/buffer_successes": i // 3, "online_rl/buffer_failures": i // 4,
             "online_rl/warmup_satisfied": 1, "online_rl/critic_only": 0,
             "online_rl/loss_critic": 1.0 / i, "online_rl/loss_actor": 2.0 / i,
             "online_rl/autonomous_success_rate": min(0.9, i / 20)}
            for i in range(1, 15)
        ])

    def test_a_single_row_renders(self, tmp_path):
        # span would be 0 without the guard in the shared-axis code.
        self._render(tmp_path, [{"step": 7, "online_rl/buffer_transitions": 10}])
