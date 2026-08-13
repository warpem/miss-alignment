"""
Plot final alignment error vs degradation level, one panel per experiment type.

Usage:
    python plot_convergence_radius.py convergence_radius_summary.json [--output out.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "noise": (
        "Gaussian jitter std added to GT alignment (pixels)",
        "Random-noise degradation",
    ),
    "interpolation": ("tiltxcorr-residual multiplier", "tiltxcorr-pattern degradation"),
    "snr": ("target image SNR", "Image-noise (thickness) degradation"),
}


def plot_summary(rows: list[dict], out_path: Path) -> None:
    by_type: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r["final_mean_error_angstrom"] is None:
            continue
        by_type.setdefault(r["type"], []).append(
            (r["level"], r["final_mean_error_angstrom"])
        )

    if not by_type:
        print("No results to plot")
        return

    types = sorted(by_type)
    fig, axes = plt.subplots(1, len(types), figsize=(5 * len(types), 4), squeeze=False)
    for ax, exp_type in zip(axes[0], types):
        points = sorted(by_type[exp_type])
        xs, ys = zip(*points)
        ax.plot(xs, ys, "o-")
        xlabel, title = LABELS.get(exp_type, (exp_type, exp_type))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final mean alignment error (Å)")
        ax.set_title(title)
        if exp_type == "snr":
            # worse (lower) SNR to the right, matching the other panels
            ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = json.loads(args.summary_json.read_text())
    plot_summary(rows, args.output or args.summary_json.with_suffix(".png"))
