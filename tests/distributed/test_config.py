import json
import pytest
from pathlib import Path
from miss_alignment.distributed.config import ClusterConfig, load_cluster_config


@pytest.fixture()
def cluster_json(tmp_path):
    cfg = {
        "submit": "sbatch {{script_path}}",
        "submit_job_id_regex": r"Submitted batch job (\d+)",
        "cancel": "scancel {{job_id}}",
    }
    p = tmp_path / "cluster.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture()
def cluster_script(tmp_path):
    p = tmp_path / "worker.sh"
    p.write_text("#!/bin/bash\n{{command}}\n")
    return p


def test_load_cluster_config_raises_when_config_unset(monkeypatch):
    monkeypatch.delenv("MISS_CLUSTER_CONFIG", raising=False)
    monkeypatch.delenv("MISS_CLUSTER_SCRIPT", raising=False)
    with pytest.raises(RuntimeError, match="MISS_CLUSTER_CONFIG"):
        load_cluster_config()


def test_load_cluster_config_raises_when_script_unset(monkeypatch, cluster_json):
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(cluster_json))
    monkeypatch.delenv("MISS_CLUSTER_SCRIPT", raising=False)
    with pytest.raises(RuntimeError, match="MISS_CLUSTER_SCRIPT"):
        load_cluster_config()


def test_load_cluster_config_returns_config(monkeypatch, cluster_json, cluster_script):
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(cluster_json))
    monkeypatch.setenv("MISS_CLUSTER_SCRIPT", str(cluster_script))
    cfg = load_cluster_config()
    assert isinstance(cfg, ClusterConfig)
    assert "sbatch" in cfg.submit
    assert cfg.script_path == cluster_script
    assert r"(\d+)" in cfg.submit_job_id_regex


def test_load_cluster_config_raises_on_missing_file(monkeypatch, tmp_path, cluster_script):
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(tmp_path / "nonexistent.json"))
    monkeypatch.setenv("MISS_CLUSTER_SCRIPT", str(cluster_script))
    with pytest.raises(FileNotFoundError):
        load_cluster_config()


def test_load_cluster_config_raises_on_missing_key(monkeypatch, tmp_path, cluster_script):
    bad = tmp_path / "bad.json"
    bad.write_text('{"submit": "sbatch {{script_path}}"}')
    monkeypatch.setenv("MISS_CLUSTER_CONFIG", str(bad))
    monkeypatch.setenv("MISS_CLUSTER_SCRIPT", str(cluster_script))
    with pytest.raises(KeyError):
        load_cluster_config()
