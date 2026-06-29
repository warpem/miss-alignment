"""Tests for inference mode (``infer.py``).

These do not require a GPU: ``run_alignment_parallel`` is monkeypatched so the
test exercises only the orchestration -- selecting the right per-iteration model
from a previous run, snapshotting results, and recording provenance.
"""

import json
from pathlib import Path

import pytest
import yaml

from miss_alignment import infer as infer_module
from miss_alignment.infer import infer_miss_align


def _write_config(tmp_path, data_dir, model_run_dir, iteration_settings):
    config = {
        "general": {
            "data_directory": str(data_dir),
            "model_run_directory": str(model_run_dir),
            "apply_ctf": False,
            "iteration_settings": iteration_settings,
            "seed": 0,
        },
        "tilt_series_alignment": {
            "patch_size": 96,
            "patch_overlap": 0.1,
            "batch_size": 32,
        },
    }
    config_path = tmp_path / "infer.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _make_models(model_run_dir, n):
    """Create iter1..iterN/model.ckpt stubs as a finished run would."""
    for i in range(1, n + 1):
        iteration_directory = model_run_dir / f"iter{i}"
        iteration_directory.mkdir(parents=True)
        (iteration_directory / "model.ckpt").write_text("ckpt")


def test_infer_applies_models_per_iteration(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ts1.xml").write_text("xml1")
    (data_dir / "ts2.xml").write_text("xml2")

    model_run_dir = tmp_path / "run"
    iteration_settings = [
        {"downsample": 3, "alignment": "anchoring"},
        {"downsample": 1, "alignment": [3, 3]},
    ]
    _make_models(model_run_dir, len(iteration_settings))
    config_path = _write_config(tmp_path, data_dir, model_run_dir, iteration_settings)

    calls = []

    def fake_run_alignment_parallel(
        *,
        model_checkpoint,
        tilt_series_list,
        output_directory,
        setting,
        downsample,
        **_,
    ):
        calls.append(
            {
                "model_checkpoint": model_checkpoint,
                "setting": setting,
                "downsample": downsample,
            }
        )
        # mimic the real writer: produce one loss json per tilt series
        for ts in tilt_series_list:
            (Path(output_directory) / f"{ts.stem}_alignment_loss.json").write_text(
                json.dumps([0.0])
            )

    monkeypatch.setattr(
        infer_module, "run_alignment_parallel", fake_run_alignment_parallel
    )

    infer_miss_align(
        config_file=config_path,
        start_at_iteration=0,
        prepare_stacks=None,
        preprocess=False,
    )

    # one alignment call per iteration, in order, with the matching model + setting
    assert len(calls) == 2
    assert calls[0]["model_checkpoint"] == str(model_run_dir / "iter1" / "model.ckpt")
    assert calls[0]["setting"] == "anchoring"
    assert calls[0]["downsample"] == 3
    assert calls[1]["model_checkpoint"] == str(model_run_dir / "iter2" / "model.ckpt")
    assert calls[1]["setting"] == [3, 3]
    assert calls[1]["downsample"] == 1

    # snapshots created per iteration, with provenance recorded
    for i in range(1, len(iteration_settings) + 1):
        iter_dir = data_dir / f"iter{i}"
        assert (iter_dir / "ts1.xml").exists()
        assert (iter_dir / "ts1_alignment_loss.json").exists()
        source = (iter_dir / "model_source.txt").read_text().strip()
        assert source == str((model_run_dir / f"iter{i}" / "model.ckpt").resolve())


def test_infer_missing_model_raises_before_alignment(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ts1.xml").write_text("xml1")

    model_run_dir = tmp_path / "run"
    iteration_settings = [
        {"downsample": 3, "alignment": "anchoring"},
        {"downsample": 1, "alignment": "global"},
    ]
    # only the first iteration's model exists
    _make_models(model_run_dir, 1)
    config_path = _write_config(tmp_path, data_dir, model_run_dir, iteration_settings)

    called = []
    monkeypatch.setattr(
        infer_module,
        "run_alignment_parallel",
        lambda **kwargs: called.append(kwargs),
    )

    with pytest.raises(FileNotFoundError, match="iteration 2"):
        infer_miss_align(
            config_file=config_path,
            start_at_iteration=0,
            prepare_stacks=None,
            preprocess=False,
        )

    # validation happens up front, before any alignment work
    assert called == []
