"""
Run the full SHREC convergence-radius experiment suite end-to-end.

For every condition in settings.yaml (see README.md): generate the degraded
input if needed, run `miss-alignment train` on it, score the result against
ground truth with compare_to_ground_truth.py's own code, and finally collect
everything into one summary table + plot.

Usage:
    python run_experiment.py [--settings settings.yaml]
    python run_experiment.py --skip-generate       # conditions already built
    python run_experiment.py --skip-generate --skip-train  # only re-score/aggregate
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # examples/shrec

from generate_conditions import main as generate_conditions, resolve_device  # noqa: E402
from degradation import list_models  # noqa: E402
from compare_to_ground_truth import calculate_alignment_error  # noqa: E402
from miss_alignment.data.io import TiltSeriesData  # noqa: E402


def run_training(cond: dict, settings: dict) -> bool:
    """Run `miss-alignment train` for one condition. Returns True on success."""
    cond_dir = Path(cond["dir"])
    n_iters = len(settings["train_config"]["general"]["iteration_settings"])
    final_ckpt = cond_dir / f"iter{n_iters}" / "model.ckpt"
    if final_ckpt.exists():
        print(f"[{cond['name']}] already trained, skipping")
        return True

    run_cfg = settings["run"]
    cmd = [
        "miss-alignment",
        "train",
        "--config-file",
        str(cond_dir / "config.yaml"),
        "--training-devices",
        str(run_cfg.get("training_devices", "0")),
        "--reconstruction-devices",
        str(run_cfg.get("reconstruction_devices", "0")),
        "--dataloaders-per-trainer",
        str(run_cfg.get("dataloaders_per_trainer", 2)),
        "--pool-size",
        str(run_cfg.get("pool_size", 1000)),
    ]
    env = os.environ.copy()
    env.update({k: str(v) for k, v in run_cfg.get("env", {}).items()})

    log_path = cond_dir / "train.log"
    print(f"[{cond['name']}] running: {' '.join(cmd)} (log: {log_path})")
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"[{cond['name']}] FAILED (exit {result.returncode}), see {log_path}")
        return False
    print(f"[{cond['name']}] training done")
    return True


def evaluate_condition(
    cond: dict, settings: dict, raw_data_dir: Path, models: list[str], device: str
) -> dict:
    """Score every requested iterK/ snapshot of a condition against ground truth."""
    cond_dir = Path(cond["dir"])
    n_iters = len(settings["train_config"]["general"]["iteration_settings"])
    track_all = settings["eval"].get("track_all_iterations")
    iters_to_eval = range(n_iters + 1) if track_all else [n_iters]

    trajectory = {}
    for it in iters_to_eval:
        it_dir = cond_dir / f"iter{it}"
        if not it_dir.is_dir():
            continue
        per_model = {}
        for model in models:
            test_xml = it_dir / f"{model}.xml"
            if not test_xml.exists():
                continue
            gt_data = TiltSeriesData(
                xml_metadata_path=raw_data_dir / "ground_truth" / f"{model}.xml"
            )
            test_data = TiltSeriesData(xml_metadata_path=test_xml)
            result = calculate_alignment_error(gt_data, test_data, device)
            per_model[model] = result["mean_error_angstrom"]
        if per_model:
            trajectory[f"iter{it}"] = {
                "per_model_mean_error_angstrom": per_model,
                "mean_error_angstrom": sum(per_model.values()) / len(per_model),
            }

    (cond_dir / "eval_result.json").write_text(json.dumps(trajectory, indent=2))
    return trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings", type=Path, default=Path(__file__).parent / "settings.yaml"
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Assume conditions already generated",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (e.g. to only re-score and re-aggregate existing runs)",
    )
    args = parser.parse_args()

    settings = yaml.safe_load(args.settings.read_text())
    raw_data_dir = Path(settings["raw_data_dir"]).expanduser().resolve()
    output_root = Path(settings["output_root"]).expanduser().resolve()

    if args.skip_generate:
        manifest = json.loads((output_root / "manifest.json").read_text())
    else:
        manifest = generate_conditions(args.settings)

    models = list_models(raw_data_dir, settings.get("models"))
    eval_device = resolve_device(settings["eval"].get("device", "cpu"))

    summary_rows = []
    for cond in manifest:
        if not args.skip_train:
            ok = run_training(cond, settings)
            if not ok:
                if settings["run"].get("stop_on_error", False):
                    print("Stopping suite: stop_on_error is true")
                    break
                continue

        trajectory = evaluate_condition(
            cond, settings, raw_data_dir, models, eval_device
        )
        final_key = (
            max(trajectory, key=lambda k: int(k.replace("iter", "")))
            if trajectory
            else None
        )
        summary_rows.append(
            {
                "name": cond["name"],
                "type": cond["type"],
                "level": cond["level"],
                "final_mean_error_angstrom": (
                    trajectory[final_key]["mean_error_angstrom"] if final_key else None
                ),
            }
        )

    summary_json = output_root / "convergence_radius_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2))
    with open(output_root / "convergence_radius_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "type", "level", "final_mean_error_angstrom"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote summary to {summary_json}")

    try:
        from plot_convergence_radius import plot_summary

        plot_summary(summary_rows, output_root / "convergence_radius.png")
    except Exception as e:
        print(f"Plotting failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
