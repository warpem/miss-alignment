"""Tests for inference mode (``infer.py``).

These do not require a GPU: ``run_alignment_parallel`` is monkeypatched so the
test exercises only the orchestration -- selecting the right per-iteration model
from a previous run, snapshotting results, and recording provenance.
"""

import json
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from miss_alignment import cli
from miss_alignment import infer as infer_module
from miss_alignment.infer import infer_miss_align


def _write_config(tmp_path, data_dir, model_run_dir, iteration_settings):
    config = {
        "general": {
            "data_directory": str(data_dir),
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
    if model_run_dir is not None:
        config["general"]["model_run_directory"] = str(model_run_dir)
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


def _record_alignment_calls(monkeypatch):
    """Patch alignment with a stub that records calls and writes loss jsons."""
    calls = []

    def fake_run_alignment_parallel(
        *, model_checkpoint, tilt_series_list, output_directory, **kwargs
    ):
        calls.append({"model_checkpoint": model_checkpoint, **kwargs})
        for ts in tilt_series_list:
            (Path(output_directory) / f"{ts.stem}_alignment_loss.json").write_text(
                json.dumps([0.0])
            )

    monkeypatch.setattr(
        infer_module, "run_alignment_parallel", fake_run_alignment_parallel
    )
    return calls


def _use_fake_pretrained(tmp_path, monkeypatch):
    """Point the pretrained-model lookup at local stubs instead of downloading."""
    pretrained_dir = tmp_path / "pretrained"
    _make_models(pretrained_dir, infer_module.N_PRETRAINED_ITERATIONS)
    monkeypatch.setattr(infer_module, "get_pretrained_models", lambda: pretrained_dir)
    return pretrained_dir


def test_infer_without_config_uses_pretrained_models(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ts1.xml").write_text("xml1")
    pretrained_dir = _use_fake_pretrained(tmp_path, monkeypatch)
    calls = _record_alignment_calls(monkeypatch)

    infer_miss_align(
        config_file=None,
        data_directory=data_dir,
        model_run_directory=None,
        start_at_iteration=0,
        prepare_stacks=None,
        preprocess=False,
    )

    # the full pretrained schedule runs, each step with its own model
    schedule = infer_module.PRETRAINED_ITERATION_SETTINGS
    assert len(calls) == len(schedule)
    for i, (call, setting) in enumerate(zip(calls, schedule), start=1):
        assert call["model_checkpoint"] == str(
            pretrained_dir / f"iter{i}" / "model.ckpt"
        )
        assert call["setting"] == setting["alignment"]
        assert call["downsample"] == setting["downsample"]
        assert call["patch_size"] == 96
    assert "easymode dataset by Mart So-Last" in capsys.readouterr().out


def test_infer_cli_model_run_directory_overrides_pretrained(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ts1.xml").write_text("xml1")
    model_run_dir = tmp_path / "run"
    _make_models(model_run_dir, 1)
    config_path = _write_config(
        tmp_path, data_dir, None, [{"downsample": 1, "alignment": "global"}]
    )
    monkeypatch.setattr(
        infer_module,
        "get_pretrained_models",
        lambda: pytest.fail("pretrained models must not be used"),
    )
    calls = _record_alignment_calls(monkeypatch)

    infer_miss_align(
        config_file=config_path,
        data_directory=None,
        model_run_directory=model_run_dir,
        start_at_iteration=0,
        prepare_stacks=None,
        preprocess=False,
    )

    assert [c["model_checkpoint"] for c in calls] == [
        str(model_run_dir / "iter1" / "model.ckpt")
    ]


def test_infer_pretrained_custom_schedule_warns(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "ts1.xml").write_text("xml1")
    _use_fake_pretrained(tmp_path, monkeypatch)
    calls = _record_alignment_calls(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        infer_module.logger, "warning", lambda msg, *args: warnings.append(msg)
    )
    custom = [{"downsample": 1, "alignment": "global"}]
    config_path = _write_config(tmp_path, data_dir, None, custom)

    infer_miss_align(
        config_file=config_path,
        data_directory=None,
        model_run_directory=None,
        start_at_iteration=0,
        prepare_stacks=None,
        preprocess=False,
    )

    # other schedules are allowed, but flagged
    assert len(calls) == 1
    assert len(warnings) == 1


def test_infer_pretrained_schedule_prefix_does_not_warn(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        infer_module.logger, "warning", lambda msg, *args: warnings.append(msg)
    )
    # tuples (as from Python) and lists (as from YAML) compare equal
    prefix = [
        {"downsample": 3, "alignment": "anchoring"},
        {"downsample": 2, "alignment": "anchoring"},
    ]
    infer_module._check_pretrained_schedule(prefix)
    infer_module._check_pretrained_schedule(
        infer_module.PRETRAINED_ITERATION_SETTINGS[:4]
        + [{"downsample": 1, "alignment": (3, 3)}]
    )
    assert warnings == []


def test_infer_pretrained_too_many_iterations_raises():
    too_long = infer_module.PRETRAINED_ITERATION_SETTINGS + [
        {"downsample": 1, "alignment": "global"}
    ]
    with pytest.raises(ValueError, match="only cover"):
        infer_module._check_pretrained_schedule(too_long)


def test_infer_requires_data_directory(tmp_path, monkeypatch):
    _use_fake_pretrained(tmp_path, monkeypatch)
    with pytest.raises(typer.BadParameter, match="--data-directory"):
        infer_miss_align(
            config_file=None,
            data_directory=None,
            model_run_directory=None,
            start_at_iteration=0,
            prepare_stacks=None,
            preprocess=False,
        )


def test_infer_model_run_directory_requires_schedule(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(ValueError, match="iteration_settings"):
        infer_miss_align(
            config_file=None,
            data_directory=data_dir,
            model_run_directory=tmp_path / "run",
            start_at_iteration=0,
            prepare_stacks=None,
            preprocess=False,
        )


def _make_stub_dataset(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "ts.xml").write_text("working XML")
    # This test covers orchestration; real XML parsing is covered elsewhere.
    monkeypatch.setattr(
        infer_module, "exclude_nonfinite_alignment_tilt_series", lambda *_: []
    )
    return data


@pytest.mark.parametrize("start", [0, 1])
def test_legacy_positional_call_with_all_existing_options(tmp_path, monkeypatch, start):
    data = _make_stub_dataset(tmp_path, monkeypatch)
    models = tmp_path / "models"
    _make_models(models, 2)
    config = _write_config(
        tmp_path, data, models, [{"downsample": 1, "alignment": "global"}] * 2
    )
    if start:
        (data / "iter1").mkdir()
        (data / "iter1" / "ts.xml").write_text("snapshot XML")
    calls = _record_alignment_calls(monkeypatch)
    prepared, preprocessed = [], []
    monkeypatch.setattr(
        infer_module, "prepare_stacks_parallel", lambda **kw: prepared.append(kw)
    )
    monkeypatch.setattr(
        infer_module,
        "run_cross_correlation_alignment_parallel",
        lambda **kw: preprocessed.append(kw),
    )

    infer_module.infer_miss_align(config, start, 10.0, start == 0, 2)

    assert len(calls) == 2 - start
    assert calls[0]["model_checkpoint"] == str(
        models / f"iter{start + 1}" / "model.ckpt"
    )
    assert all(c["n_cluster_workers"] == 2 for c in calls)
    assert prepared[0]["desired_pixel_size"] == 10.0
    assert prepared[0]["n_cluster_workers"] == 2
    assert len(preprocessed) == (1 if start == 0 else 0)
    assert (data / "iter2" / "ts.xml").read_text() == (
        "snapshot XML" if start else "working XML"
    )


@pytest.mark.parametrize("entrypoint", ["python", "cli"])
def test_no_config_pretrained_run(tmp_path, monkeypatch, entrypoint):
    data = _make_stub_dataset(tmp_path, monkeypatch)
    models = _use_fake_pretrained(tmp_path, monkeypatch)
    calls = _record_alignment_calls(monkeypatch)
    if entrypoint == "python":
        infer_module.infer_miss_align(
            data_directory=data,
            start_at_iteration=0,
            prepare_stacks=None,
            preprocess=False,
            n_cluster_workers=None,
        )
    else:
        result = CliRunner().invoke(cli, ["infer", "--data-directory", str(data)])
        assert result.exit_code == 0, (result.output, result.exception)
    assert len(calls) == 8
    for i, call in enumerate(calls, 1):
        assert call["model_checkpoint"] == str(models / f"iter{i}" / "model.ckpt")
        assert (
            call["setting"]
            == infer_module.PRETRAINED_ITERATION_SETTINGS[i - 1]["alignment"]
        )
        assert (
            call["downsample"]
            == infer_module.PRETRAINED_ITERATION_SETTINGS[i - 1]["downsample"]
        )
        assert call["n_cluster_workers"] is None


def test_cli_overrides_and_resume(tmp_path, monkeypatch):
    data = _make_stub_dataset(tmp_path, monkeypatch)
    (data / "iter1").mkdir()
    (data / "iter1" / "ts.xml").write_text("restored XML")
    models = tmp_path / "models"
    _make_models(models, 2)
    config = _write_config(
        tmp_path,
        tmp_path / "wrong-data",
        tmp_path / "wrong-models",
        [{"downsample": 1, "alignment": "global"}] * 2,
    )
    calls = _record_alignment_calls(monkeypatch)
    prepared = []
    monkeypatch.setattr(
        infer_module,
        "get_pretrained_models",
        lambda: pytest.fail("should use explicit models"),
    )
    monkeypatch.setattr(
        infer_module, "prepare_stacks_parallel", lambda **kw: prepared.append(kw)
    )
    result = CliRunner().invoke(
        cli,
        [
            "infer",
            "--config-file",
            str(config),
            "--data-directory",
            str(data),
            "--model-run-directory",
            str(models),
            "--start-at-iteration",
            "1",
            "--prepare-stacks",
            "10",
            "--n-cluster-workers",
            "2",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(calls) == 1
    assert calls[0]["model_checkpoint"] == str(models / "iter2" / "model.ckpt")
    assert calls[0]["n_cluster_workers"] == 2
    assert prepared[0]["desired_pixel_size"] == 10.0
    assert (data / "iter2" / "ts.xml").read_text() == "restored XML"
    assert not (tmp_path / "wrong-data").exists()
