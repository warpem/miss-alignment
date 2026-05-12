# miss-alignment

[![License](https://img.shields.io/pypi/l/miss-alignment.svg?color=green)](https://github.com/McHaillet/miss-alignment/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/miss-alignment.svg?color=green)](https://pypi.org/project/miss-alignment)
[![Python Version](https://img.shields.io/pypi/pyversions/miss-alignment.svg?color=green)](https://python.org)
[![CI](https://github.com/McHaillet/miss-alignment/actions/workflows/ci.yml/badge.svg)](https://github.com/McHaillet/miss-alignment/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/McHaillet/miss-alignment/branch/main/graph/badge.svg)](https://codecov.io/gh/McHaillet/miss-alignment)

## Installation

Installation is limited at the moment to a specific python, CUDA, and torch version. This might be fixed at some point in the future. For now, its easiest to set everything up in a conda environment.

First create an environment called `miss-alignment` with cuda-toolkit 12.9 and activate it:

```
conda create –n miss-alignment –c conda-forge python=3.11 cuda-toolkit=12.9 –y
conda activate miss-alignment
```

We need to fix some GPU dependencies for accelerated reconstruction:
```
python -m pip install torch==2.8.0 numpy
python -m pip install torch-projectors --index-url https://warpem.github.io/torch-projectors/cu129/simple/
```

Finally install miss-alignment with this command:

```
python -m pip install miss-alignment
```

Check that the CLI shows up with:

```
miss-alignment --help
```

## How to run?

See the docs/ folder for some barebones instructions. This will be improved soon.

## Changelog

A full list of changes per release is available on the [GitHub Releases page](https://github.com/McHaillet/miss-alignment/releases).
