"""Pretrained miss-alignment models from easymode.

The models were trained by Mart So-Last on 54 datasets as part of easymode
(https://mgflast.github.io/easymode/, https://github.com/mgflast/easymode) and are
distributed via Hugging Face under GPL-3.0. They are downloaded on first use and
cached, never bundled with miss-alignment.

The download is pinned to a Hugging Face commit and verified against a SHA-256
checksum: the checkpoints are unpickled by Lightning, so only a known archive is
ever unpacked. Updating the models means bumping these constants, which also
changes the cache directory so earlier results stay reproducible.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ._cli import cli

PRETRAINED_REVISION = "d4e9a07ef0ea2695c05ebd7b0e2c49cb50d3da26"
PRETRAINED_URL = (
    "https://huggingface.co/mgflast/easymode-v2/resolve/"
    f"{PRETRAINED_REVISION}/missalignment/easymode_missalignment.zip"
)
PRETRAINED_SHA256 = "7cae259abf7c8c11c90e7a5ae3a588101c5ce1b21d6abd8710e4761e1527b43b"
ARCHIVE_ROOT = "easymode_missalignment"

# Schedule the pretrained models were trained for: model iterN expects step N.
PRETRAINED_ITERATION_SETTINGS = [
    {"downsample": 3, "alignment": "anchoring"},
    {"downsample": 2, "alignment": "anchoring"},
    {"downsample": 1, "alignment": "global"},
    {"downsample": 1, "alignment": "global"},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
    {"downsample": 1, "alignment": [3, 3]},
]
N_PRETRAINED_ITERATIONS = len(PRETRAINED_ITERATION_SETTINGS)

ATTRIBUTION = (
    "Pretrained models from easymode by Mart So-Last "
    "(https://mgflast.github.io/easymode/, GPL-3.0).\n"
    "If you use them, please also cite the easymode preprint: "
    "https://doi.org/10.64898/2026.05.19.726344"
)

# Overrides the cache location, e.g. to a shared filesystem for cluster workers.
CACHE_DIR_ENV_VAR = "MISS_ALIGNMENT_CACHE_DIR"


def default_cache_dir() -> Path:
    """Cache root: ``$MISS_ALIGNMENT_CACHE_DIR``, else the XDG cache directory."""
    if os.environ.get(CACHE_DIR_ENV_VAR):
        return Path(os.environ[CACHE_DIR_ENV_VAR])
    xdg_cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(xdg_cache) / "miss-alignment"


def pretrained_model_dir(cache_dir: Optional[Path] = None) -> Path:
    """Directory holding ``iter1..iterN/model.ckpt`` for the pinned revision."""
    cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return cache_dir / "models" / f"easymode-{PRETRAINED_REVISION[:7]}"


def _is_complete(model_dir: Path) -> bool:
    return all(
        (model_dir / f"iter{i}" / "model.ckpt").is_file()
        for i in range(1, N_PRETRAINED_ITERATIONS + 1)
    )


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0) or None
        with (
            open(destination, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="download") as bar,
        ):
            while chunk := response.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def get_pretrained_models(
    cache_dir: Optional[Path] = None,
    url: str = PRETRAINED_URL,
    sha256: str = PRETRAINED_SHA256,
) -> Path:
    """Return the cached pretrained model directory, downloading it if needed.

    Once cached, no network access happens. The archive is downloaded and
    unpacked into a temporary directory next to the cache and only moved into
    place when complete, so an interrupted download never leaves a partial
    cache and concurrent runs cannot corrupt each other.
    """
    model_dir = pretrained_model_dir(cache_dir)
    if _is_complete(model_dir):
        return model_dir

    print(f"Downloading pretrained models to {model_dir}")
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=model_dir.parent, prefix=".tmp-") as tmp:
        tmp = Path(tmp)
        archive = tmp / "models.zip"
        _download(url, archive)

        actual_sha256 = _sha256(archive)
        if actual_sha256 != sha256:
            raise RuntimeError(
                f"Checksum mismatch for pretrained models downloaded from {url}: "
                f"expected {sha256}, got {actual_sha256}."
            )

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp / "extracted")
        extracted = tmp / "extracted" / ARCHIVE_ROOT
        if not _is_complete(extracted):
            raise RuntimeError(
                f"Pretrained model archive from {url} does not contain "
                f"iter1..iter{N_PRETRAINED_ITERATIONS}/model.ckpt."
            )

        # a stale incomplete directory (e.g. manually damaged) is replaced
        if model_dir.exists() and not _is_complete(model_dir):
            shutil.rmtree(model_dir)
        try:
            extracted.rename(model_dir)
        except OSError:
            # another process finished the same download first
            if not _is_complete(model_dir):
                raise

    return model_dir


@cli.command(name="download-models")
def download_models() -> None:
    """Download the pretrained easymode models used by `infer` by default.

    Useful on clusters where compute nodes have no internet access: run this
    once on a login node. The cache must be on a filesystem visible to the
    nodes that run `infer`. The cache defaults to ~/.cache/miss-alignment; set
    MISS_ALIGNMENT_CACHE_DIR to place it elsewhere (for both commands).
    """
    model_dir = get_pretrained_models()
    print(ATTRIBUTION)
    print(f"Pretrained models available at: {model_dir}")
