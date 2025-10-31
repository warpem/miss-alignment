"""She has a chaotic good alignment for tilt-series."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("miss_alignment")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Marten Chaillet"
__email__ = "martenchaillet@gmail.com"
__all__ = [
    "__version__",
    "cli",
    "download_training_data",
    "train_miss_align",
]

from ._cli import cli
from .data import download_training_data
from .train import train_miss_align
