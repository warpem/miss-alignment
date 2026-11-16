"""Tests for downloading and caching the pretrained models (``pretrained.py``).

No network access: a small fake archive is served from a ``file://`` URL.
"""

import hashlib
import zipfile

import pytest

from miss_alignment import pretrained
from miss_alignment.pretrained import (
    ARCHIVE_ROOT,
    N_PRETRAINED_ITERATIONS,
    get_pretrained_models,
    pretrained_model_dir,
)


def _make_archive(path, n_iterations=N_PRETRAINED_ITERATIONS):
    """Write a zip laid out like the easymode archive; return (url, sha256)."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{ARCHIVE_ROOT}/inference_config.yaml", "general: {}\n")
        for i in range(1, n_iterations + 1):
            zf.writestr(f"{ARCHIVE_ROOT}/iter{i}/model.ckpt", f"ckpt{i}")
    return path.as_uri(), hashlib.sha256(path.read_bytes()).hexdigest()


def test_download_then_reuse_cache(tmp_path, monkeypatch):
    url, sha256 = _make_archive(tmp_path / "models.zip")
    cache_dir = tmp_path / "cache"

    model_dir = get_pretrained_models(cache_dir, url=url, sha256=sha256)

    assert model_dir == pretrained_model_dir(cache_dir)
    for i in range(1, N_PRETRAINED_ITERATIONS + 1):
        assert (model_dir / f"iter{i}" / "model.ckpt").read_text() == f"ckpt{i}"
    # only the final directory is left behind, no temporary download dirs
    assert list(model_dir.parent.iterdir()) == [model_dir]

    def fail_download(*_):
        raise AssertionError("cached models must not be downloaded again")

    monkeypatch.setattr(pretrained, "_download", fail_download)
    assert get_pretrained_models(cache_dir, url=url, sha256=sha256) == model_dir


def test_checksum_mismatch_leaves_no_cache(tmp_path):
    url, _ = _make_archive(tmp_path / "models.zip")
    cache_dir = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        get_pretrained_models(cache_dir, url=url, sha256="0" * 64)

    model_dir = pretrained_model_dir(cache_dir)
    assert not model_dir.exists()
    assert list(model_dir.parent.iterdir()) == []


def test_incomplete_archive_raises(tmp_path):
    url, sha256 = _make_archive(tmp_path / "models.zip", n_iterations=3)
    cache_dir = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="does not contain"):
        get_pretrained_models(cache_dir, url=url, sha256=sha256)

    assert not pretrained_model_dir(cache_dir).exists()


def test_incomplete_cache_is_replaced(tmp_path):
    url, sha256 = _make_archive(tmp_path / "models.zip")
    cache_dir = tmp_path / "cache"
    model_dir = pretrained_model_dir(cache_dir)
    (model_dir / "iter1").mkdir(parents=True)  # e.g. a manually damaged cache

    assert get_pretrained_models(cache_dir, url=url, sha256=sha256) == model_dir
    assert (model_dir / f"iter{N_PRETRAINED_ITERATIONS}" / "model.ckpt").is_file()


def test_cache_dir_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(pretrained.CACHE_DIR_ENV_VAR, str(tmp_path / "shared"))
    assert pretrained.default_cache_dir() == tmp_path / "shared"
    assert pretrained_model_dir().is_relative_to(tmp_path / "shared")

    monkeypatch.delenv(pretrained.CACHE_DIR_ENV_VAR)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert pretrained.default_cache_dir() == tmp_path / "xdg" / "miss-alignment"
