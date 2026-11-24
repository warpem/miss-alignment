"""
Build all degraded-input SHREC project directories from settings.yaml.

For every enabled experiment type and sweep value, this creates
``<output_root>/<condition_name>/*.xml`` (+ symlinked or noised .st stacks)
directly in the condition directory -- the ``training_directory`` for
``miss-alignment train``, which expects its input .xml files flat at its
root and creates ``iter0/``, ``iter1/``, ... itself as backup snapshots
during training. It also writes a ``config.yaml`` and, at the end,
``<output_root>/manifest.json`` listing every condition. Safe to re-run:
already-generated tilt-series are left untouched, but each condition's
config.yaml is always rewritten from the current settings.

Usage:
    python generate_conditions.py [--settings settings.yaml]
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from degradation import (  # noqa: E402
    SHREC_DIR,
    compute_tiltxcorr_residuals,
    condition_seed,
    fmt_level,
    generate_interpolation_condition,
    generate_noise_condition,
    generate_snr_condition,
    list_models,
)


def ensure_raw_data(raw_data_dir: Path) -> None:
    """Download + prepare the SHREC benchmark if it isn't there yet.

    A directory that already has ground_truth/ and iter0/ xml files (e.g.
    already downloaded, or prepared some other way) is left completely
    untouched -- preproc.py's conversion step is not idempotent against
    hand-modified xmls, so it is only invoked for a genuinely fresh download.
    """
    gt_dir = raw_data_dir / "ground_truth"
    it0_dir = raw_data_dir / "iter0"
    if (
        gt_dir.exists()
        and it0_dir.exists()
        and any(gt_dir.glob("*.xml"))
        and any(it0_dir.glob("*.xml"))
    ):
        return

    sys.path.insert(0, str(SHREC_DIR))
    import preproc

    preproc.download_and_unzip(raw_data_dir)
    preproc.convert_pickles_to_xml(raw_data_dir)


def write_condition_config(cond_dir: Path, settings: dict) -> None:
    config = copy.deepcopy(settings["train_config"])
    config["general"]["training_directory"] = str(cond_dir)
    config["general"]["seed"] = settings["seed"]
    cond_dir.mkdir(parents=True, exist_ok=True)
    with open(cond_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def resolve_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def main(settings_path: Path) -> list[dict]:
    settings = yaml.safe_load(settings_path.read_text())
    raw_data_dir = Path(settings["raw_data_dir"]).expanduser().resolve()
    output_root = Path(settings["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ensure_raw_data(raw_data_dir)
    models = list_models(raw_data_dir, settings.get("models"))
    if not models:
        raise ValueError(f"No models found under {raw_data_dir / 'ground_truth'}")
    seed = settings["seed"]
    exp = settings["experiments"]

    manifest = []

    if exp.get("noise", {}).get("enabled"):
        for std in exp["noise"]["std_values_pixels"]:
            name = f"noise_std{fmt_level(std)}"
            cond_dir = output_root / name
            for model in models:
                out_xml = cond_dir / f"{model}.xml"
                if out_xml.exists():
                    continue
                seed_i = condition_seed(seed, name, model)
                generate_noise_condition(model, std, raw_data_dir, out_xml, seed_i)
            write_condition_config(cond_dir, settings)
            manifest.append(
                {"name": name, "type": "noise", "level": std, "dir": str(cond_dir)}
            )
            print(f"[generate] {name}: {len(models)} tilt-series ready")

    if exp.get("interpolation", {}).get("enabled"):
        device = resolve_device(settings["eval"].get("device", "cpu"))
        residuals = compute_tiltxcorr_residuals(
            raw_data_dir, models, device, output_root / "tiltxcorr_residuals.json"
        )
        for mult in exp["interpolation"]["multipliers"]:
            name = f"interp_{fmt_level(mult)}x"
            cond_dir = output_root / name
            for model in models:
                out_xml = cond_dir / f"{model}.xml"
                if out_xml.exists():
                    continue
                generate_interpolation_condition(
                    model, mult, raw_data_dir, out_xml, residuals
                )
            write_condition_config(cond_dir, settings)
            manifest.append(
                {
                    "name": name,
                    "type": "interpolation",
                    "level": mult,
                    "dir": str(cond_dir),
                }
            )
            print(f"[generate] {name}: {len(models)} tilt-series ready")

    if exp.get("snr", {}).get("enabled"):
        for thickness_nm in exp["snr"]["thickness_values_nm"]:
            name = f"snr_{fmt_level(thickness_nm)}nm"
            cond_dir = output_root / name
            for model in models:
                out_xml = cond_dir / f"{model}.xml"
                if out_xml.exists():
                    continue
                seed_i = condition_seed(seed, name, model)
                generate_snr_condition(
                    model, thickness_nm, raw_data_dir, out_xml, seed_i
                )
            write_condition_config(cond_dir, settings)
            manifest.append(
                {
                    "name": name,
                    "type": "snr",
                    "level": thickness_nm,
                    "dir": str(cond_dir),
                }
            )
            print(f"[generate] {name}: {len(models)} tilt-series ready")

    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} conditions to {output_root / 'manifest.json'}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings", type=Path, default=Path(__file__).parent / "settings.yaml"
    )
    args = parser.parse_args()
    main(args.settings)
