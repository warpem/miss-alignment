"""
Plot final alignment error vs degradation level, one panel per experiment
type, showing the per-SHREC-model spread (swarm plot) behind the median/mean.

Companion to plot_convergence_radius.py: that script draws the mean-only
line plot that run_experiment.py generates automatically from
convergence_radius_summary.json. This one also needs each condition's
eval_result.json (written next to config.yaml, not copied into the summary
files) for the per-model breakdown, so point it at the same output_root /
results directory that holds both the summary files and the condition
subdirectories (each containing eval_result.json).

Usage:
    python plot_convergence_radius_spread.py RESULTS_DIR [--output out.png]
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from degradation import SHREC_DIR, SNR0, snr_at_thickness  # noqa: E402
from plot_convergence_radius import LABELS  # noqa: E402

if str(SHREC_DIR) not in sys.path:
    sys.path.insert(0, str(SHREC_DIR))
from preproc import ORIGINAL_PIXEL_SIZE  # noqa: E402

# Panel order matches settings.yaml's `experiments:` section rather than
# alphabetical, so panels read noise -> interpolation -> snr.
TYPE_ORDER = ["noise", "interpolation", "snr"]

# noise_std16/32 are already fully degraded (same order-of-magnitude error as
# noise_std8) and only stretch the axis without showing anything new, so cap
# the noise panel at 8 std.
MAX_NOISE_LEVEL = 8


def load_spread(results_dir: Path) -> pd.DataFrame:
    """Collect per-SHREC-model final errors for every condition in the summary."""
    summary = json.loads((results_dir / "convergence_radius_summary.json").read_text())
    rows = []
    for r in summary:
        cond_dir = results_dir / "experiments" / r["name"]
        if not cond_dir.is_dir():
            cond_dir = results_dir / r["name"]
        eval_path = cond_dir / "eval_result.json"
        if not eval_path.exists():
            print(f"[skip] no eval_result.json for {r['name']}")
            continue
        trajectory = json.loads(eval_path.read_text())
        final_key = max(trajectory, key=lambda k: int(k.replace("iter", "")))
        per_model = trajectory[final_key]["per_model_mean_error_angstrom"]
        for model, error in per_model.items():
            rows.append(
                {
                    "type": r["type"],
                    "level": r["level"],
                    "model": model,
                    "error_angstrom": error,
                }
            )
    df = pd.DataFrame(rows)
    df = df[(df["type"] != "noise") | (df["level"] <= MAX_NOISE_LEVEL)]
    # 1.5x sits awkwardly close to 1x/2x and doesn't add to the story.
    df = df[~((df["type"] == "interpolation") & (df["level"] == 1.5))]
    # 900nm sits close to 1100nm and doesn't add to the story now that
    # 1500nm shows the breakdown more clearly.
    df = df[~((df["type"] == "snr") & (df["level"] == 900))]

    # The snr sweep's nominal "target thickness" is a simplistic proxy --
    # annotate with the model's predicted SNR instead (see degradation.py).
    snr_mask = df["type"] == "snr"
    df.loc[snr_mask, "level"] = df.loc[snr_mask, "level"].apply(snr_at_thickness)

    # Add the zero-added-noise baseline (SNR0, at the reference thickness):
    # the snr sweep itself doesn't run this condition (it's plain tiltxcorr,
    # already produced by the interpolation sweep's 1x condition, which
    # reproduces tiltxcorr's own alignment from ground truth).
    baseline = df[(df["type"] == "interpolation") & (df["level"] == 1)].copy()
    baseline["type"] = "snr"
    baseline["level"] = SNR0
    df = pd.concat([df, baseline], ignore_index=True)

    return df


def load_interp_sigma(results_dir: Path) -> float | None:
    """Std (Angstrom) of the 1x tiltxcorr-residual magnitude, pooled over all
    tilts and models -- the spread of the initial (pre-training) alignment
    error the interpolation sweep injects at multiplier=1. Scales linearly
    with the multiplier for the other levels (the residual is just scaled,
    not redrawn, see generate_interpolation_condition in degradation.py).
    """
    path = results_dir / "tiltxcorr_residuals.json"
    if not path.exists():
        print(f"[skip] no {path.name}, interpolation x-ticks won't show sigma")
        return None
    residuals = json.loads(path.read_text())
    magnitudes = [
        math.hypot(y, x)
        for r in residuals.values()
        for y, x in zip(r["y_error_angstrom"], r["x_error_angstrom"])
    ]
    return statistics.stdev(magnitudes)


def plot_spread(
    df: pd.DataFrame,
    out_path: Path,
    stat: str = "median",
    log_y: bool = True,
    interp_sigma: float | None = None,
) -> None:
    types = [t for t in TYPE_ORDER if t in df["type"].unique()]
    fig, axes = plt.subplots(
        1, len(types), figsize=(4.5 * len(types), 4.5), sharey=True,
        layout="constrained",
    )
    if len(types) == 1:
        axes = [axes]

    for ax, exp_type in zip(axes, types):
        sub = df[df["type"] == exp_type]
        ax.set_box_aspect(1)
        # yscale must be set before swarmplot() computes non-overlap offsets,
        # otherwise it spaces points assuming a linear axis and they collide
        # once the log scale is applied afterwards.
        if log_y:
            ax.set_yscale("log")
        sns.swarmplot(
            data=sub,
            x="level",
            y="error_angstrom",
            ax=ax,
            size=6.4,
            color="#4c72b0",
            alpha=0.6,
            native_scale=True,
            zorder=1,
        )
        grouped = sub.groupby("level")["error_angstrom"]
        if stat == "median":
            center = grouped.median().sort_index()
            q1 = grouped.quantile(0.25).sort_index()
            q3 = grouped.quantile(0.75).sort_index()
            yerr = [center.values - q1.values, q3.values - center.values]
            stat_label = "median (error bars: IQR)"
        else:
            center = grouped.mean().sort_index()
            std = grouped.std().sort_index()
            yerr = std.values
            stat_label = "mean (error bars: std)"
        ax.errorbar(
            center.index,
            center.values,
            yerr=yerr,
            fmt="D-",
            color="black",
            markersize=6,
            linewidth=1.5,
            capsize=3,
            zorder=3,
        )
        xlabel, title = LABELS.get(exp_type, (exp_type, exp_type))
        if exp_type == "snr":
            # Override the shared LABELS entry (thickness-based, used by the
            # mean-only plot_convergence_radius.py): the thickness->SNR
            # conversion is too simplistic to report thickness itself.
            xlabel, title = "Approximate image SNR", "Image-noise (SNR) degradation"
        elif exp_type == "noise":
            # Override the shared LABELS entry (pixels): report sigma only
            # in Angstrom, matching the interpolation panel's annotation.
            xlabel = "Gaussian jitter starting alignment in σ (Å)"
        elif exp_type == "interpolation":
            xlabel = "tiltxcorr alignment multiplier: factor, σ (Å)"
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        levels = sorted(sub["level"].unique())
        ax.set_xticks(levels)
        if exp_type == "snr":
            # 3 decimals + rotation: the low-SNR end is now crowded (0.08,
            # 0.04, 0.03) and 2-decimal labels were indistinguishable/overlapping.
            ax.set_xticklabels(
                [f"{lv:.3f}" for lv in levels], rotation=45, ha="right"
            )
            # Higher SNR = less noise = less degraded; flip so degradation
            # still increases left-to-right, matching the other two panels.
            ax.invert_xaxis()
        elif exp_type == "noise":
            ax.set_xticklabels(
                [f"{lv * ORIGINAL_PIXEL_SIZE:.0f}" for lv in levels]
            )
        elif exp_type == "interpolation" and interp_sigma is not None:
            # Each tick: multiplier factor, std of the initial (pre-training)
            # alignment error it injects (multiplier * sigma of the
            # underlying 1x tiltxcorr residual).
            ax.set_xticklabels(
                [f"{lv:g}x, {lv * interp_sigma:.0f}" for lv in levels]
            )

    axes[0].set_ylabel("Final mean alignment error (Å)")
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#4c72b0",
            alpha=0.6, markersize=6, label="per-model error (n=10 SHREC volumes)",
        ),
        Line2D(
            [0], [0], marker="D", color="black", markersize=6, linewidth=1.5,
            label=stat_label,
        ),
    ]
    axes[0].legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=8)

    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory with convergence_radius_summary.json and per-condition "
        "eval_result.json files (output_root, or a local mirror of it)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--stat",
        choices=["median", "mean"],
        default="median",
        help="Central tendency + error bars for the line: median+IQR or mean+std",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of the default log scale",
    )
    args = parser.parse_args()

    data = load_spread(args.results_dir)
    interp_sigma = load_interp_sigma(args.results_dir)
    out = args.output or args.results_dir / "convergence_radius_spread.png"
    plot_spread(
        data, out, stat=args.stat, log_y=not args.linear_y, interp_sigma=interp_sigma
    )
