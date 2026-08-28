"""Plot the RL-token reconstruction loss from a run's ``losses.json``.

    python diagnostics/plot_rl_token_loss.py crp_rl_token outputs/rl_token_report

The convergence question -- "has the reconstruction loss flattened?" -- is what
gates stage 3, and a single log-scale curve answers it badly: the first 1000
steps compress everything after them. So the right panel plots the per-window
mean instead, where a plateau is a plateau.

Reuses the palette and mark specs from plot_train_curves.py.
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_train_curves import THEME, _end_label, _plain_log_ticks, _style  # noqa: E402

WINDOW = 200  # rolling median over the raw per-step loss
BLOCK = 1000  # per-window mean for the convergence panel


def plot(losses, meta, outdir, mode):
    c = THEME[mode]
    steps = np.arange(len(losses))
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1))
    fig.patch.set_facecolor(c["surface"])

    gamma = meta.get("norm_gamma", 0.0)
    ntok = meta.get("num_rl_tokens", "?")
    fig.suptitle(
        "RL token — reconstruction loss", color=c["ink"], fontsize=15,
        fontweight="bold", x=0.012, ha="left", y=0.975,
    )
    fig.text(
        0.012, 0.900,
        f"CRP dual-arm · pi0.5 frozen · {len(losses):,} steps · {ntok} RL token, 2048-dim"
        f" · per-dim weighted MSE (gamma={gamma})",
        color=c["muted"], fontsize=9.5, ha="left",
    )

    # --- left: the curve itself ---
    a = axes[0]
    a.plot(steps, losses, color=c["s1"], linewidth=0.6, alpha=0.20, zorder=2)
    med = np.array([
        np.median(losses[max(0, i - WINDOW // 2): i + WINDOW // 2 + 1]) for i in range(len(losses))
    ])
    a.plot(steps, med, color=c["s1"], linewidth=1.8, zorder=3)
    a.set_yscale("log")
    imin = int(np.argmin(losses))
    a.plot(imin, losses[imin], marker="o", markersize=7, color=c["s1"],
           markeredgecolor=c["surface"], markeredgewidth=2, zorder=5)
    # Corner note rather than an anchored label: by the time the minimum lands the
    # curve is flat, so anything attached to that point sits on top of the line.
    a.annotate(
        f"min {losses[imin]:.4f} @ step {imin:,}",
        xy=(0.97, 0.90), xycoords="axes fraction", ha="right",
        color=c["muted"], fontsize=8.5,
    )
    _end_label(a, c, steps[-1], med[-1], f"{med[-1]:.4f}", c["s1"])
    a.set_ylim(0.18, 5.0)
    _plain_log_ticks(a, subs=(1.0, 2.0, 3.0, 5.0))
    a.set_xlim(0, len(losses) * 1.06)
    _style(a, c, xlabel="step", ylabel="weighted MSE (log)", title="Loss",
           subtitle=f"faint = per step, solid = {WINDOW}-step rolling median")

    # --- right: is it actually flat? ---
    a = axes[1]
    nblocks = len(losses) // BLOCK
    xs = np.arange(1, nblocks + 1) * BLOCK
    means = np.array([np.mean(losses[i * BLOCK:(i + 1) * BLOCK]) for i in range(nblocks)])
    a.plot(xs, means, color=c["s1"], linewidth=1.8, marker="o", markersize=6,
           markeredgecolor=c["surface"], markeredgewidth=1.5, zorder=3)
    for x, y in zip(xs, means):
        a.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 9), textcoords="offset points",
                   color=c["secondary"], fontsize=7.5, ha="center",
                   path_effects=[matplotlib.patheffects.withStroke(
                       linewidth=2.5, foreground=c["surface"])])
    # How much is the last block still buying over the one before it?
    if nblocks >= 2:
        gain = (means[-2] - means[-1]) / means[-2] * 100
        a.annotate(
            f"last 1k buys only {gain:.1f}% over the previous 1k",
            xy=(0.97, 0.90), xycoords="axes fraction", ha="right",
            color=c["muted"], fontsize=8.5,
        )
    a.set_ylim(0, max(means) * 1.22)
    a.set_xlim(0, len(losses) * 1.06)
    _style(a, c, xlabel="step", ylabel=f"mean over each {BLOCK} steps", title="Convergence",
           subtitle="linear axis — a plateau here is a real plateau")

    fig.tight_layout(rect=(0, 0, 1, 0.865))
    path = Path(outdir) / f"rl_token_loss_{mode}.png"
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)
    print("wrote", path)
    return means


def main(run_dir, outdir):
    run_dir, outdir = Path(run_dir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    losses = np.array(json.loads((run_dir / "losses.json").read_text()), dtype=float)

    meta = {}
    ckpt = run_dir / "demo_adapt_checkpoint.pt"
    if ckpt.is_file():
        import torch

        meta = torch.load(ckpt, map_location="cpu", weights_only=False).get("metadata") or {}

    means = None
    for mode in ("light", "dark"):
        means = plot(losses, meta, outdir, mode)

    lines = [
        "# RL token 重建 loss",
        "",
        f"{len(losses):,} 步 · {meta.get('num_rl_tokens', '?')} 个 RL token · "
        f"norm_gamma={meta.get('norm_gamma', 0.0)}",
        "",
        "| 区间 | 均值 | 相对前一区间 |",
        "| --- | --- | --- |",
    ]
    for i, m in enumerate(means):
        rel = "—" if i == 0 else f"−{(means[i-1]-m)/means[i-1]*100:.1f}%"
        lines.append(f"| {i*BLOCK:,}–{(i+1)*BLOCK:,} | {m:.4f} | {rel} |")
    lines += [
        "",
        f"最低单点 {losses.min():.4f} @ 步 {int(np.argmin(losses)):,}；"
        f"末 1000 步均值 {means[-1]:.4f}。",
    ]
    (outdir / "rl_token_summary.md").write_text("\n".join(lines) + "\n")
    print("wrote", outdir / "rl_token_summary.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "crp_rl_token",
         sys.argv[2] if len(sys.argv) > 2 else "outputs/rl_token_report")
