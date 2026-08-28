"""Plot the pi0.5 SFT run's metrics from the CSVs that ``wandb_offline_export.py`` writes.

    python diagnostics/plot_train_curves.py outputs/pi05_vla_ft_report

Writes ``training_<mode>.png`` and ``hardware_<mode>.png`` for both light and dark,
plus ``summary.md`` (the table view -- identity is never carried by color alone).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

# Palette slots 1-2 of the validated default categorical theme, used in fixed order.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "s1": "#2a78d6",
        "s2": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "s1": "#3987e5",
        "s2": "#d95926",
    },
}

# rolling window over ~7.5 s system samples, i.e. a 2 minute median
SYS_WINDOW = 16


def _plain_log_ticks(ax, subs=(1.0, 2.0, 3.0, 5.0)):
    """Label a log y-axis with plain decimals -- 0.3 reads faster than 3x10^-1."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=20))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _style(ax, c, *, xlabel, ylabel, title, subtitle=None):
    ax.set_facecolor(c["surface"])
    ax.grid(axis="y", color=c["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["baseline"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c["muted"], labelsize=8.5, length=3, width=0.8)
    ax.set_xlabel(xlabel, color=c["muted"], fontsize=8.5)
    ax.set_ylabel(ylabel, color=c["secondary"], fontsize=9)
    head = title if subtitle is None else f"{title}\n"
    ax.set_title(head, color=c["ink"], fontsize=11, fontweight="semibold", loc="left", pad=10)
    if subtitle:
        ax.annotate(
            subtitle,
            xy=(0, 1.012),
            xycoords="axes fraction",
            color=c["muted"],
            fontsize=8.5,
            ha="left",
            va="bottom",
        )


def _end_label(ax, c, x, y, text, color, dy=0):
    """Direct label riding the end of a line, ringed by the surface so it stays legible."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(4, dy),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        fontweight="semibold",
        va="center",
        ha="left",
        path_effects=[pe.withStroke(linewidth=2.5, foreground=c["surface"])],
        zorder=6,
    )


def _legend(ax, c, loc="upper right"):
    leg = ax.legend(frameon=False, fontsize=8.5, loc=loc, handlelength=1.6)
    for txt in leg.get_texts():
        txt.set_color(c["secondary"])


def _figure(nrows, ncols, mode, suptitle, subtitle):
    c = THEME[mode]
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.3 * nrows))
    fig.patch.set_facecolor(c["surface"])
    fig.suptitle(suptitle, color=c["ink"], fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.952, subtitle, color=c["muted"], fontsize=9.5, ha="left")
    return fig, axes.ravel(), c


def plot_training(hist, outdir, mode):
    c = THEME[mode]
    fig, ax, c = _figure(
        2, 2, mode,
        "pi0.5 SFT — training metrics",
        "local/crp_rlt_dataset · 622 ep / 220,967 frames @ 16 fps · batch 16 · 30,000 steps (2.17 epochs) · 51.3 h",
    )
    step = hist["_step"]

    # --- loss (single series: the title names it, so no legend box) ---
    a = ax[0]
    a.plot(step, hist["train/loss"], color=c["s1"], linewidth=1.8, zorder=3)
    a.set_yscale("log")
    imin = hist["train/loss"].idxmin()
    a.plot(
        hist.loc[imin, "_step"], hist.loc[imin, "train/loss"],
        marker="o", markersize=7, color=c["s1"], markeredgecolor=c["surface"], markeredgewidth=2, zorder=5,
    )
    a.annotate(
        f"min {hist.loc[imin, 'train/loss']:.4f} @ {int(hist.loc[imin, '_step']):,}",
        xy=(hist.loc[imin, "_step"], hist.loc[imin, "train/loss"]),
        xytext=(0, -16), textcoords="offset points",
        color=c["secondary"], fontsize=8.5, ha="center",
        path_effects=[pe.withStroke(linewidth=2.5, foreground=c["surface"])],
    )
    _end_label(a, c, step.iloc[-1], hist["train/loss"].iloc[-1], f"{hist['train/loss'].iloc[-1]:.4f}", c["s1"])
    a.set_ylim(0.0055, 0.30)
    _plain_log_ticks(a)
    _style(a, c, xlabel="step", ylabel="flow-matching loss (log)", title="Training loss",
           subtitle="mean over each 200-step logging window")

    # --- gradient norm ---
    a = ax[1]
    a.plot(step, hist["train/grad_norm"], color=c["s1"], linewidth=1.8, zorder=3)
    a.set_yscale("log")
    a.axhline(1.0, color=c["baseline"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    a.annotate("clip norm = 1.0", xy=(step.iloc[-1], 1.0), xytext=(-4, 5), textcoords="offset points",
               color=c["muted"], fontsize=8, ha="right")
    _end_label(a, c, step.iloc[-1], hist["train/grad_norm"].iloc[-1],
               f"{hist['train/grad_norm'].iloc[-1]:.3f}", c["s1"])
    _plain_log_ticks(a)
    _style(a, c, xlabel="step", ylabel="grad norm, pre-clip (log)", title="Gradient norm",
           subtitle="settles just under a third of the clip threshold")

    # --- learning rate ---
    a = ax[2]
    a.plot(step, hist["train/lr"] * 1e5, color=c["s1"], linewidth=1.8, zorder=3)
    a.axvline(1000, color=c["baseline"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    a.annotate("warmup ends\n1,000", xy=(1000, 0.35), xytext=(6, 0), textcoords="offset points",
               color=c["muted"], fontsize=8, ha="left", va="center")
    a.set_ylim(0, 2.85)
    _end_label(a, c, step.iloc[-1], hist["train/lr"].iloc[-1] * 1e5, "0.25", c["s1"])
    _style(a, c, xlabel="step", ylabel=r"learning rate  ($\times 10^{-5}$)", title="Learning rate",
           subtitle="1k linear warmup to 2.5e-5, cosine decay to 2.5e-6")

    # --- step time: two series, same unit, one axis (log) ---
    a = ax[3]
    a.plot(step, hist["train/update_s"], color=c["s1"], linewidth=1.8, label="optimizer update", zorder=3)
    a.plot(step, hist["train/dataloading_s"], color=c["s2"], linewidth=1.8, label="data loading", zorder=3)
    a.set_yscale("log")
    _end_label(a, c, step.iloc[-1], hist["train/update_s"].iloc[-1],
               f"{hist['train/update_s'].iloc[-1]:.2f}s", c["s1"])
    _end_label(a, c, step.iloc[-1], hist["train/dataloading_s"].iloc[-1],
               f"{hist['train/dataloading_s'].iloc[-1]:.3f}s", c["s2"])
    a.set_ylim(0.012, 12)
    _plain_log_ticks(a, subs=(1.0,))
    _style(a, c, xlabel="step", ylabel="seconds per step (log)", title="Step time",
           subtitle="compute-bound: the 8 dataloader workers are ~300x ahead of the GPU")
    _legend(a, c, loc="center left")

    for a in ax:
        a.set_xlim(0, 31600)
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))

    fig.tight_layout(rect=(0, 0, 1, 0.938))
    path = Path(outdir) / f"training_{mode}.png"
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)
    print("wrote", path)


def _sys_pair(a, c, sysdf, key0, key1, labels):
    """GPU 1 underneath, GPU 0 on top -- GPU 0 is this run, GPU 1 is a co-tenant job."""
    layers = ((key1, c["s2"], labels[1], 2), (key0, c["s1"], labels[0], 4))
    handles = {}
    for key, color, label, z in layers:
        s = sysdf[["hours", key]].dropna()
        a.plot(s["hours"], s[key], color=color, linewidth=0.6, alpha=0.15, zorder=z - 1)
        med = s[key].rolling(SYS_WINDOW, min_periods=1, center=True).median()
        (line,) = a.plot(s["hours"], med, color=color, linewidth=1.6, label=label, zorder=z)
        handles[label] = line
    return [handles[labels[0]], handles[labels[1]]]


def _threshold(a, c, y, text):
    a.axhline(y, color=c["baseline"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    a.annotate(
        text,
        xy=(1.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 4),
        textcoords="offset points",
        color=c["muted"],
        fontsize=8,
        ha="right",
        path_effects=[pe.withStroke(linewidth=2.5, foreground=c["surface"])],
        zorder=6,
    )


def plot_hardware(sysdf, outdir, mode):
    fig, ax, c = _figure(
        3, 2, mode,
        "pi0.5 SFT — training box telemetry",
        "2x RTX A6000 48 GiB · 51.3 h wall clock · faint trace = 7.5 s samples, solid = 2 min rolling median",
    )
    gpu_labels = ("GPU 0 — this run", "GPU 1 — co-tenant job")

    a = ax[0]
    handles = _sys_pair(a, c, sysdf, "gpu.0.gpu", "gpu.1.gpu", gpu_labels)
    a.set_ylim(-3, 108)
    _style(a, c, xlabel="elapsed hours", ylabel="SM utilization (%)", title="GPU utilization",
           subtitle="GPU 0 pinned at 99.3% mean — the run never waits on data")

    a = ax[1]
    _sys_pair(a, c, sysdf, "gpu.0.memoryAllocated", "gpu.1.memoryAllocated", gpu_labels)
    a.set_ylim(-3, 108)
    _style(a, c, xlabel="elapsed hours", ylabel="VRAM allocated (%)", title="GPU memory",
           subtitle="39.0 GiB of 48 GiB on GPU 0 at batch 16 with gradient checkpointing")

    a = ax[2]
    _sys_pair(a, c, sysdf, "gpu.0.temp", "gpu.1.temp", gpu_labels)
    a.set_ylim(52, 95)
    _threshold(a, c, 90, "90 °C slowdown threshold")
    _style(a, c, xlabel="elapsed hours", ylabel="core temperature (°C)", title="GPU temperature",
           subtitle="GPU 0 sat at 88.7 °C mean and touched 90 °C — thermally capped")

    a = ax[3]
    _sys_pair(a, c, sysdf, "gpu.0.powerWatts", "gpu.1.powerWatts", gpu_labels)
    a.set_ylim(0, 330)
    _threshold(a, c, 300, "300 W enforced limit")
    _style(a, c, xlabel="elapsed hours", ylabel="board power (W)", title="GPU power draw",
           subtitle="GPU 0 held 287.7 W mean, i.e. 95.9% of its cap")

    a = ax[4]
    _sys_pair(a, c, sysdf, "gpu.0.smClock", "gpu.1.smClock", gpu_labels)
    a.set_ylim(150, 2050)
    _threshold(a, c, 1860, "1860 MHz peak observed")
    _style(a, c, xlabel="elapsed hours", ylabel="SM clock (MHz)", title="GPU clock",
           subtitle="GPU 0 averaged 1537 MHz — the thermal/power cap costs ~17% of peak clock")

    a = ax[5]
    host = _sys_pair(a, c, sysdf, "cpu", "memory_percent", ("host CPU", "host RAM"))
    a.set_ylim(0, 20)
    _style(a, c, xlabel="elapsed hours", ylabel="host utilization (%)", title="Host CPU and RAM",
           subtitle="2.1% of 96 threads, 7.4% of 503 GiB — the host is nowhere near the bottleneck")
    leg = a.legend(handles=host, frameon=False, fontsize=8.5, loc="upper right", handlelength=1.6)
    for txt in leg.get_texts():
        txt.set_color(c["secondary"])

    for a in ax:
        a.set_xlim(0, 51.8)

    # One shared legend for the GPU 0 / GPU 1 pairing rather than six colliding boxes.
    leg = fig.legend(
        handles=handles, labels=list(gpu_labels), loc="upper left",
        bbox_to_anchor=(0.010, 0.938), frameon=False, fontsize=9.5, ncol=2, handlelength=1.6,
        columnspacing=1.8,
    )
    for txt in leg.get_texts():
        txt.set_color(c["secondary"])

    fig.tight_layout(rect=(0, 0, 1, 0.930))
    path = Path(outdir) / f"hardware_{mode}.png"
    fig.savefig(path, dpi=200, facecolor=c["surface"])
    plt.close(fig)
    print("wrote", path)


def write_summary(hist, sysdf, outdir):
    lines = [
        "# pi0.5 SFT run `biy4hbcy` — metric summary",
        "",
        "Table view of the same numbers the PNGs plot, so nothing depends on color.",
        "",
        "## Training (150 logging windows, every 200 steps)",
        "",
        "| metric | @200 | @10k | @20k | @30k | min | max |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    idx = {s: hist.index[hist["_step"] == s][0] for s in (200, 10000, 20000, 30000)}
    for key, label, fmt in (
        ("train/loss", "loss", "{:.4f}"),
        ("train/grad_norm", "grad norm", "{:.3f}"),
        ("train/lr", "learning rate", "{:.2e}"),
        ("train/update_s", "update s/step", "{:.3f}"),
        ("train/dataloading_s", "dataloading s/step", "{:.4f}"),
        ("train/epochs", "epochs", "{:.2f}"),
    ):
        cells = [fmt.format(hist.loc[idx[s], key]) for s in (200, 10000, 20000, 30000)]
        lines.append(
            f"| {label} | " + " | ".join(cells)
            + f" | {fmt.format(hist[key].min())} | {fmt.format(hist[key].max())} |"
        )

    lines += [
        "",
        "## Hardware (24,650 samples over 51.3 h)",
        "",
        "| metric | mean | min | max |",
        "| --- | --- | --- | --- |",
    ]
    for key, label, fmt in (
        ("gpu.0.gpu", "GPU 0 utilization (%)", "{:.1f}"),
        ("gpu.0.memoryAllocated", "GPU 0 VRAM (%)", "{:.1f}"),
        ("gpu.0.temp", "GPU 0 temp (°C)", "{:.1f}"),
        ("gpu.0.powerWatts", "GPU 0 power (W)", "{:.1f}"),
        ("gpu.0.smClock", "GPU 0 SM clock (MHz)", "{:.0f}"),
        ("gpu.1.gpu", "GPU 1 utilization (%)", "{:.1f}"),
        ("gpu.1.memoryAllocated", "GPU 1 VRAM (%)", "{:.1f}"),
        ("cpu", "host CPU (%)", "{:.2f}"),
        ("memory_percent", "host RAM (%)", "{:.2f}"),
        ("proc.memory.rssMB", "trainer RSS (MB)", "{:.0f}"),
    ):
        s = sysdf[key].dropna()
        lines.append(
            f"| {label} | {fmt.format(s.mean())} | {fmt.format(s.min())} | {fmt.format(s.max())} |"
        )

    path = Path(outdir) / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    print("wrote", path)


def main(outdir):
    out = Path(outdir)
    hist = pd.read_csv(out / "history.csv").sort_values("_step").reset_index(drop=True)
    sysdf = pd.read_csv(out / "system.csv").sort_values("_timestamp").reset_index(drop=True)
    sysdf["hours"] = (sysdf["_timestamp"] - sysdf["_timestamp"].min()) / 3600.0
    for mode in ("light", "dark"):
        plot_training(hist, out, mode)
        plot_hardware(sysdf, out, mode)
    write_summary(hist, sysdf, out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/pi05_vla_ft_report")
