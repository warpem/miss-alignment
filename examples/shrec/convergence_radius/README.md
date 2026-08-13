# SHREC convergence-radius experiments

Reviewer-response experiments: how far from the correct alignment can
miss-alignment start and still converge back to it? Three ways of pushing the
SHREC benchmark's starting alignment further from ground truth, each swept
over a severity parameter, each run through the *full* iterative
train+realign pipeline (`miss-alignment train`) from scratch per condition.

| Experiment      | Starting point           | What's degraded                                   | Swept over |
|-----------------|---------------------------|----------------------------------------------------|------------|
| `noise`         | ground truth              | random-normal per-tilt jitter added to offsets     | jitter std, pixels |
| `interpolation` | ground truth              | a multiple of the ground-truth→tiltxcorr residual  | multiplier (1x = tiltxcorr) |
| `snr`           | tiltxcorr (iter0)         | Gaussian noise added to the images                 | target SNR |

Only translations (`tilt_axis_offset_x/y`) are ever perturbed; angles and
tilt axis angle always come from the source unchanged (tiltxcorr does not
touch them for this dataset, and neither does any degradation used here).

## One command

```bash
conda activate miss-torch26   # or wherever miss-alignment is installed
python run_experiment.py
```

This will, on a system that has nothing yet:
1. Download and prepare the SHREC benchmark into `raw_data_dir` (via
   `../preproc.py`), if it isn't already there.
2. Build every condition's degraded `iter0/` + `config.yaml` under
   `output_root` (skipping any tilt-series that already exist there).
3. Run `miss-alignment train` for each condition in turn (skips conditions
   whose final checkpoint already exists, so a killed run can be restarted
   with the same command).
4. Score every finished condition against ground truth using the same code
   `compare_to_ground_truth.py` already uses.
5. Write `convergence_radius_summary.{json,csv}` and a
   `convergence_radius.png` plot under `output_root`.

Edit `settings.yaml` to change anything -- sweep values, training
hyperparameters, devices, or which stages run. See the comments in that file.

Useful partial runs:
```bash
python generate_conditions.py            # only build the degraded inputs
python run_experiment.py --skip-generate # assume conditions already built
python run_experiment.py --skip-generate --skip-train  # only re-score + re-plot
```

## Notes / assumptions worth checking

- **`raw_data_dir` already populated is never re-derived.** If
  `ground_truth/` and `iter0/` already contain `.xml` files (as with an
  existing hand-processed download), `run_experiment.py` leaves them
  completely alone rather than re-running `preproc.py`'s pickle→xml
  conversion, since that step isn't idempotent against any local edits.
  Delete/rename the directory (or point `raw_data_dir` elsewhere) to force a
  clean re-download.
- **The tiltxcorr residual removes tiltxcorr's global 3D offset first.**
  Raw ground-truth-vs-tiltxcorr offsets differ by 10s-100s of Å per tilt in a
  tilt-angle-dependent way -- this is an unresolved global 3D translation
  (tomographic alignment has no absolute origin), not real misalignment.
  `interpolation` removes it via `compare_to_ground_truth.calculate_alignment_error`
  (cross-correlation between reconstructed volumes) before scaling the
  residual by the multiplier, so 1x/2x/3x actually escalate tiltxcorr's error
  pattern rather than a coordinate-frame artifact. This computation runs once
  per model and is cached to `tiltxcorr_residuals.json`.
- **SNR definition**: `SNR = Var(clean image stack) / Var(added Gaussian noise)`,
  computed per tilt-series from the (already roughly zero-mean, unit-std)
  ground-truth image stack.
- **Every condition trains from random initial weights** (`model_checkpoint:
  null`), i.e. the full method (model training + alignment, both iterated)
  runs independently per condition -- this is deliberately expensive; there's
  no shortcut of reusing one fixed trained model here. Point
  `model_training.model_checkpoint` at a `.ckpt` to warm-start every
  condition from the same weights instead of random init.
- Conditions run **sequentially**, one `miss-alignment train` at a time,
  using the devices configured under `run:` in `settings.yaml`.
