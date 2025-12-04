# miss-alignment 👩‍🔧

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

Then clone the github repository and install the software:

```
git clone https://github.com/warpem/miss-alignment.git
python -m pip install miss-alignment
```
