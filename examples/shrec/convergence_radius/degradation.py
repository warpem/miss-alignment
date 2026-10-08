"""
Shared helpers for building degraded-alignment SHREC input series.

Three degradation modes are supported, all starting from the ground-truth
alignment (or, for the SNR mode, the tiltxcorr alignment):

* ``noise``         -- ground truth + random-normal per-tilt jitter, swept
                        over a standard deviation (pixels).
* ``interpolation``  -- ground truth + a multiple of the (global-offset
                        corrected) ground-truth-to-tiltxcorr residual, swept
                        over a multiplier (1x = tiltxcorr itself).
* ``snr``            -- tiltxcorr alignment, images degraded by added
                        Gaussian noise, swept over target sample thickness
                        (nm); the target SNR at each thickness follows the
                        exponential decay model calibrated at the top of this
                        file (see ``snr_at_thickness``).

Only translations (``tilt_axis_offset_x/y``) are ever perturbed -- angles and
tilt axis angle are always carried over unchanged from the source, matching
how the rest of miss-alignment's synthetic shift generation only touches
translations (see ``miss_alignment.data.shift_generation``).
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import mrcfile
import torch
from torch_affine_utils.transforms_3d import Ry, Rz
from warpylib import TiltSeries

from miss_alignment.data.shift_generation import project_shifts_3d_to_2d

SHREC_DIR = Path(__file__).resolve().parent.parent
if str(SHREC_DIR) not in sys.path:
    sys.path.insert(0, str(SHREC_DIR))

# --- SNR-vs-thickness calibration ---------------------------------------
# The source images for the `snr` experiment already carry real noise at
# SNR0, measured at the reference thickness T0_NM -- they are not noise-free.
# SNR is modelled to decay exponentially with sample thickness:
#   SNR(t) = SNR0 * exp(-(t - T0_NM) / LAMBDA_EFF_NM)
# LAMBDA_EFF_NM is a placeholder pending calibration against real data.
SNR0 = 0.35
T0_NM = 180.0
LAMBDA_EFF_NM = 350.0


def snr_at_thickness(
    t: float, SNR0: float = SNR0, t0: float = T0_NM, lambda_eff: float = LAMBDA_EFF_NM
) -> float:
    """Predicted SNR at sample thickness ``t`` (nm), per the decay model above."""
    return SNR0 * math.exp(-(t - t0) / lambda_eff)


def noise_std_for_target(images: torch.Tensor, SNR0: float, SNR_target: float) -> float:
    """Std of Gaussian noise to add so ``images`` (already noisy) reach SNR_target.

    Assumes ``images`` already contains noise at SNR0, not a noise-free
    signal: Var(images) = Var(signal) + Var(existing noise), with
    SNR0 = Var(signal) / Var(existing noise). Solving for the extra noise
    variance needed to bring the total down to SNR_target gives:
        noise_std = images.std() * sqrt( (1/(1+SNR0)) * (SNR0/SNR_target - 1) )
    """
    return float(images.std()) * math.sqrt(
        (1.0 / (1.0 + SNR0)) * (SNR0 / SNR_target - 1.0)
    )


def list_models(raw_data_dir: Path, explicit: list[str] | None = None) -> list[str]:
    """All model names to run, e.g. ``model_0``. Explicit list overrides."""
    if explicit:
        return list(explicit)
    return sorted(p.stem for p in (raw_data_dir / "ground_truth").glob("*.xml"))


def condition_seed(base_seed: int, *parts: str) -> int:
    """Deterministic per-condition/model seed, stable across platforms/runs."""
    key = "|".join([str(base_seed), *parts]).encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def fmt_level(x: float) -> str:
    """Compact string for a sweep value, used in condition directory names."""
    return f"{x:g}"


def projection_matrices_for(tilt_series: TiltSeries) -> torch.Tensor:
    """Per-tilt (2, 3) matrices projecting a 3D ZYX shift to a 2D YX image shift."""
    r0 = Ry(-tilt_series.angles, zyx=True)
    r1 = Rz(tilt_series.tilt_axis_angles, zyx=True)
    rotation_matrices = r1 @ r0
    return rotation_matrices[..., 1:3, :3]


def build_degraded_tilt_series(
    source_xml: Path,
    out_xml: Path,
    offset_y: torch.Tensor,
    offset_x: torch.Tensor,
    stack_source_st: Path,
    image_snr: float | None = None,
    rng: torch.Generator | None = None,
) -> None:
    """Write a new tilt-series XML with the given offsets at ``out_xml``.

    Angles, tilt axis angle, and volume/image dimensions are copied from
    ``source_xml``. The underlying image stack is either symlinked from
    ``stack_source_st`` unchanged (``image_snr is None``), or rewritten with
    added Gaussian noise so the combined (existing + added) noise brings it
    down to ``image_snr`` -- see ``noise_std_for_target``, which assumes
    ``stack_source_st`` already carries SNR0 noise rather than being clean.
    """
    ts = TiltSeries(source_xml)
    ts.path = str(out_xml)
    ts.tilt_axis_offset_y = offset_y.clone().float()
    ts.tilt_axis_offset_x = offset_x.clone().float()

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    stack_dst = Path(ts.tilt_stack_path)

    if image_snr is not None:
        with mrcfile.open(stack_source_st) as mrc:
            images = torch.tensor(mrc.data, dtype=torch.float32)
            pixel_size = float(mrc.voxel_size.x)
        noise_std = noise_std_for_target(images, SNR0, image_snr)
        noise = torch.empty_like(images).normal_(mean=0.0, std=noise_std, generator=rng)
        ts.stack_tilts(images + noise, pixel_size, create_thumbnails=False)
    else:
        stack_dst.parent.mkdir(parents=True, exist_ok=True)
        rawtlt_src = stack_source_st.with_suffix(".rawtlt")
        rawtlt_dst = stack_dst.with_suffix(".rawtlt")
        for src, dst in ((stack_source_st, stack_dst), (rawtlt_src, rawtlt_dst)):
            if not src.exists():
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

    ts.save_meta(str(out_xml))


def generate_noise_condition(
    model: str, std_pixels: float, raw_data_dir: Path, out_xml: Path, seed: int
) -> None:
    """Ground truth + random-normal per-tilt jitter with the given std (pixels)."""
    gt_xml = raw_data_dir / "ground_truth" / f"{model}.xml"
    gt_ts = TiltSeries(gt_xml)
    with mrcfile.open(gt_ts.tilt_stack_path, header_only=True) as mrc:
        pixel_size = float(mrc.voxel_size.x)

    gen = torch.Generator().manual_seed(seed)
    shifts_3d = torch.normal(
        mean=0.0, std=float(std_pixels), size=(gt_ts.n_tilts, 3), generator=gen
    )
    shifts_2d = project_shifts_3d_to_2d(shifts_3d, projection_matrices_for(gt_ts))
    shifts_angstrom = shifts_2d * pixel_size

    offset_y = gt_ts.tilt_axis_offset_y + shifts_angstrom[:, 0]
    offset_x = gt_ts.tilt_axis_offset_x + shifts_angstrom[:, 1]

    build_degraded_tilt_series(
        gt_xml, out_xml, offset_y, offset_x, Path(gt_ts.tilt_stack_path)
    )


def compute_tiltxcorr_residuals(
    raw_data_dir: Path,
    models: list[str],
    device: str,
    cache_path: Path,
    recompute: bool = False,
) -> dict:
    """Per-tilt ground-truth-to-tiltxcorr residual, with the global 3D offset
    that tiltxcorr leaves unresolved removed.

    Reuses ``calculate_alignment_error`` from ``compare_to_ground_truth.py``
    unchanged: it reconstructs both volumes, finds the best-fit global shift
    via cross-correlation, and returns the per-tilt error with that shift
    subtracted out. Results are cached to ``cache_path`` since this involves
    full volume reconstructions and only needs to run once regardless of how
    many interpolation multipliers are swept.
    """
    if cache_path.exists() and not recompute:
        return json.loads(cache_path.read_text())

    from compare_to_ground_truth import calculate_alignment_error
    from miss_alignment.data.io import TiltSeriesData

    residuals = {}
    for model in models:
        gt_data = TiltSeriesData(
            xml_metadata_path=raw_data_dir / "ground_truth" / f"{model}.xml"
        )
        it0_data = TiltSeriesData(
            xml_metadata_path=raw_data_dir / "iter0" / f"{model}.xml"
        )
        result = calculate_alignment_error(gt_data, it0_data, device)
        residuals[model] = {
            "y_error_angstrom": result["y_error_angstrom"].tolist(),
            "x_error_angstrom": result["x_error_angstrom"].tolist(),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(residuals, indent=2))
    return residuals


def generate_interpolation_condition(
    model: str, multiplier: float, raw_data_dir: Path, out_xml: Path, residuals: dict
) -> None:
    """Ground truth + ``multiplier`` times the tiltxcorr residual (1x == tiltxcorr)."""
    gt_xml = raw_data_dir / "ground_truth" / f"{model}.xml"
    gt_ts = TiltSeries(gt_xml)

    y_res = torch.tensor(residuals[model]["y_error_angstrom"], dtype=torch.float32)
    x_res = torch.tensor(residuals[model]["x_error_angstrom"], dtype=torch.float32)

    offset_y = gt_ts.tilt_axis_offset_y + multiplier * y_res
    offset_x = gt_ts.tilt_axis_offset_x + multiplier * x_res

    build_degraded_tilt_series(
        gt_xml, out_xml, offset_y, offset_x, Path(gt_ts.tilt_stack_path)
    )


def generate_snr_condition(
    model: str, thickness_nm: float, raw_data_dir: Path, out_xml: Path, seed: int
) -> None:
    """Tiltxcorr alignment, images degraded to the SNR predicted at thickness_nm.

    The source images already carry SNR0 noise at the reference thickness
    T0_NM (see the SNR-vs-thickness calibration constants above this module);
    this adds just enough extra Gaussian noise to bring them down to the SNR
    predicted for a sample thickness_nm thick.
    """
    it0_xml = raw_data_dir / "iter0" / f"{model}.xml"
    it0_ts = TiltSeries(it0_xml)
    gt_ts = TiltSeries(raw_data_dir / "ground_truth" / f"{model}.xml")

    target_snr = snr_at_thickness(thickness_nm)
    gen = torch.Generator().manual_seed(seed)
    build_degraded_tilt_series(
        it0_xml,
        out_xml,
        it0_ts.tilt_axis_offset_y,
        it0_ts.tilt_axis_offset_x,
        Path(gt_ts.tilt_stack_path),
        image_snr=target_snr,
        rng=gen,
    )
