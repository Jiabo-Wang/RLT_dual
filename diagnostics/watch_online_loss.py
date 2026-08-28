"""Live view of an online-RL run: tail metrics.jsonl, redraw a PNG, print a summary.

    python diagnostics/watch_online_loss.py outputs/crp_online_rl_v1/metrics.jsonl

Open the PNG it writes (same directory, ``metrics_live.png``) in VS Code and
leave it open -- the editor reloads an image when the file changes, so a
periodically rewritten PNG *is* a live curve, with no display server, no wandb
login, and nothing to keep alive but this loop.

    --once      render a single frame and exit (for a cron/CI check)
    --interval  seconds between redraws (default 20)

The panels answer the three questions worth interrupting a session for:

  1. Are the critic and actor losses going anywhere, or is the actor's loss flat
     because it is doing pure BC? (Both on one log axis: same units, and the
     comparison between them is the point.)
  2. Is the buffer growing, and does it hold both successes and failures? Warmup
     will not release without at least three of each.
  3. Is the autonomous success rate moving? That is the only metric the run is
     actually for.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The annotations are in Chinese and DejaVu Sans has no CJK glyphs, which renders
# them as boxes. Prepend a CJK face when one is installed; fall back silently.
for _cjk in ("Noto Sans CJK SC", "Droid Sans Fallback", "WenQuanYi Zen Hei"):
    if any(f.name == _cjk for f in matplotlib.font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.sans-serif"] = [_cjk] + matplotlib.rcParams["font.sans-serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_train_curves import THEME, _end_label, _legend, _plain_log_ticks, _style  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    """Parse metrics.jsonl, tolerating a half-written final line."""
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # the trainer was mid-write; it will be there next tick
    return rows


def _series(rows, key):
    """(steps, values) for one metric, skipping rows that do not carry it."""
    pairs = [(r["step"], r[key]) for r in rows
             if key in r and isinstance(r[key], (int, float))]
    if not pairs:
        return np.array([]), np.array([])
    steps, values = zip(*pairs)
    return np.array(steps, dtype=float), np.array(values, dtype=float)


def render(rows, out_path: Path, mode: str = "light") -> None:
    c = THEME[mode]
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 4.9))
    fig.patch.set_facecolor(c["surface"])

    last = rows[-1] if rows else {}
    fig.suptitle("Online RL — live", color=c["ink"], fontsize=15, fontweight="bold",
                 x=0.008, ha="left", y=0.975)
    phase = (
        "warmup (actor frozen, VLA driving)" if not last.get("online_rl/warmup_satisfied")
        else "critic-only (actor still frozen)" if last.get("online_rl/critic_only")
        else "online RL (actor + critic training)"
    )
    fig.text(
        0.008, 0.900,
        f"{len(rows)} 条记录 · step {int(last.get('step', 0)):,} · {phase}"
        f" · 更新于 {time.strftime('%H:%M:%S')}",
        color=c["muted"], fontsize=9.5, ha="left",
    )

    # --- 1. losses -------------------------------------------------------
    a = axes[0]
    plotted = False
    for key, colour, label in (
        ("online_rl/loss_critic", c["s1"], "critic"),
        ("online_rl/loss_actor", c["s2"], "actor"),
    ):
        x, y = _series(rows, key)
        if x.size == 0:
            continue
        a.plot(x, y, color=colour, linewidth=1.6, marker="o", markersize=3,
               markeredgewidth=0, label=label, zorder=3)
        _end_label(a, c, x[-1], y[-1], f"{y[-1]:.3g}", colour)
        plotted = True
    if plotted:
        if np.all(np.concatenate([_series(rows, k)[1] for k in
                                  ("online_rl/loss_critic", "online_rl/loss_actor")
                                  if _series(rows, k)[1].size]) > 0):
            a.set_yscale("log")
            _plain_log_ticks(a)
        _legend(a, c, loc="upper right")
    else:
        a.annotate("还没有 loss —— warmup 期间不训练", xy=(0.5, 0.5),
                   xycoords="axes fraction", ha="center", color=c["muted"], fontsize=10)
    _style(a, c, xlabel="episode", ylabel="loss", title="Critic / actor loss",
           subtitle="actor loss 长期平直 = 可能退化成纯 BC")

    # --- 2. buffer -------------------------------------------------------
    a = axes[1]
    x, y = _series(rows, "online_rl/buffer_transitions")
    if x.size:
        a.plot(x, y, color=c["s1"], linewidth=1.8, zorder=3)
        _end_label(a, c, x[-1], y[-1], f"{int(y[-1]):,}", c["s1"])
        a.axhline(512, color=c["baseline"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        a.annotate("warmup 门槛 512", xy=(1.0, 512), xycoords=("axes fraction", "data"),
                   xytext=(-2, 4), textcoords="offset points", ha="right",
                   color=c["muted"], fontsize=8)
    ns, nsucc = _series(rows, "online_rl/buffer_successes")
    nf, nfail = _series(rows, "online_rl/buffer_failures")
    note = []
    if nsucc.size:
        note.append(f"成功 {int(nsucc[-1])}")
    if nfail.size:
        note.append(f"失败 {int(nfail[-1])}")
    if note:
        a.annotate(" / ".join(note) + "  (各需 ≥3)", xy=(0.03, 0.92),
                   xycoords="axes fraction", color=c["secondary"], fontsize=9)
    _style(a, c, xlabel="episode", ylabel="transitions", title="Replay buffer",
           subtitle="四个 warmup 门槛: 5 集 / 512 transition / ≥3 成功 / ≥3 失败")

    # --- 3. what the run is for -----------------------------------------
    a = axes[2]
    drawn = False
    for key, colour, label in (
        ("online_rl/autonomous_success_rate", c["s1"], "累计"),
        ("online_rl/autonomous_success_rate_rolling_20", c["s2"], "最近 20"),
    ):
        x, y = _series(rows, key)
        if x.size == 0:
            continue
        a.plot(x, y * 100, color=colour, linewidth=1.8, marker="o", markersize=3,
               markeredgewidth=0, label=label, zorder=3)
        _end_label(a, c, x[-1], y[-1] * 100, f"{y[-1] * 100:.0f}%", colour)
        drawn = True
    a.set_ylim(-4, 104)
    if drawn:
        _legend(a, c, loc="upper left")
    else:
        a.annotate("还没有已标注的 episode", xy=(0.5, 0.5), xycoords="axes fraction",
                   ha="center", color=c["muted"], fontsize=10)
    _style(a, c, xlabel="episode", ylabel="自主成功率 (%)", title="Autonomous success",
           subtitle="不含人工干预救回的 episode")

    # One shared episode axis: the three panels are meant to be read against each
    # other ("the loss moved at the same episode the success rate did"), which a
    # per-panel autoscale quietly prevents -- losses only start after warmup, so
    # that panel would otherwise begin ~15 episodes to the right of the others.
    steps = [r["step"] for r in rows if isinstance(r.get("step"), (int, float))]
    if steps:
        span = max(max(steps) - min(steps), 1)
        for ax in axes:
            ax.set_xlim(min(steps) - span * 0.02, max(steps) + span * 0.06)

    fig.tight_layout(rect=(0, 0, 1, 0.868))
    # Keep the .png extension: matplotlib picks its writer from the suffix, and a
    # ".tmp" one is rejected. Write beside the target, then rename -- the rename is
    # atomic, so a viewer never opens a half-written frame.
    tmp = out_path.with_name(out_path.stem + ".partial.png")
    fig.savefig(tmp, dpi=140, facecolor=c["surface"])
    plt.close(fig)
    tmp.replace(out_path)


def summarize(rows) -> str:
    if not rows:
        return "还没有指标"
    last = rows[-1]

    def g(key, fmt="{:.4g}"):
        v = last.get(key)
        return fmt.format(v) if isinstance(v, (int, float)) else "—"

    return (
        f"ep {int(last.get('step', 0)):>4}  "
        f"critic {g('online_rl/loss_critic'):>9}  actor {g('online_rl/loss_actor'):>9}  "
        f"buffer {g('online_rl/buffer_transitions', '{:.0f}'):>6}  "
        f"成功/失败 {g('online_rl/buffer_successes', '{:.0f}')}/"
        f"{g('online_rl/buffer_failures', '{:.0f}')}  "
        f"自主 {g('online_rl/autonomous_success_rate', '{:.0%}')}  "
        f"更新 {g('online_rl/actual_updates', '{:.0f}')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", type=Path, help="path to metrics.jsonl")
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--mode", choices=("light", "dark"), default="light")
    args = ap.parse_args()

    out = args.metrics.parent / "metrics_live.png"
    print(f"看这个文件（VS Code 会自动刷新）: {out}")
    if not args.once:
        print(f"每 {args.interval:g}s 重画一次，Ctrl-C 退出\n")

    last_size = -1
    while True:
        if not args.metrics.exists():
            print(f"\r等待 {args.metrics} 出现…", end="", flush=True)
        else:
            size = args.metrics.stat().st_size
            if size != last_size or args.once:
                rows = read_rows(args.metrics)
                if rows:
                    render(rows, out, args.mode)
                print(f"\r{summarize(rows)}   ", end="", flush=True)
                last_size = size
        if args.once:
            print()
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return 0


if __name__ == "__main__":
    sys.exit(main())
