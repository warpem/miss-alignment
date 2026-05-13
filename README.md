# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/warpem/miss-alignment/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                 |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/miss\_alignment/\_\_init\_\_.py                  |       10 |        2 |     80% |       7-8 |
| src/miss\_alignment/\_cli.py                         |        8 |        1 |     88% |         9 |
| src/miss\_alignment/alignment/\_\_init\_\_.py        |        3 |        0 |    100% |           |
| src/miss\_alignment/alignment/correlation.py         |       32 |       32 |      0% |     1-112 |
| src/miss\_alignment/alignment/optimize\_global.py    |      185 |      121 |     35% |82-93, 103-124, 164-171, 177-190, 202-209, 215-228, 292-334, 356-365, 371-446, 452-519 |
| src/miss\_alignment/alignment/optimize\_iterative.py |       83 |       13 |     84% |67, 111, 116, 130, 168, 172-175, 179, 230-244 |
| src/miss\_alignment/alignment/optimize\_spline.py    |      117 |       89 |     24% |126-158, 181-320, 368-404 |
| src/miss\_alignment/alignment/parallel.py            |       47 |       36 |     23% |32-49, 85-158 |
| src/miss\_alignment/alignment/statistics.py          |       79 |        0 |    100% |           |
| src/miss\_alignment/alignment/tilt\_series.py        |       54 |       17 |     69% |   172-266 |
| src/miss\_alignment/alignment/utils.py               |        7 |        4 |     43% |     28-32 |
| src/miss\_alignment/data/\_\_init\_\_.py             |        2 |        0 |    100% |           |
| src/miss\_alignment/data/\_augmentation.py           |       39 |        0 |    100% |           |
| src/miss\_alignment/data/\_reconstruction\_worker.py |      144 |        0 |    100% |           |
| src/miss\_alignment/data/io.py                       |       82 |       29 |     65% |44, 48, 52, 57, 151-210 |
| src/miss\_alignment/data/shift\_generation.py        |      124 |        6 |     95% |148, 291-294, 297 |
| src/miss\_alignment/data/training\_datamodule.py     |       97 |       74 |     24% |38-52, 123-165, 169-170, 174-175, 190-252, 261, 280-301 |
| src/miss\_alignment/data/training\_dataset.py        |       34 |        0 |    100% |           |
| src/miss\_alignment/gradcam/\_\_init\_\_.py          |        0 |        0 |    100% |           |
| src/miss\_alignment/gradcam/gradcam.py               |       43 |       43 |      0% |      1-71 |
| src/miss\_alignment/models/\_\_init\_\_.py           |        4 |        0 |    100% |           |
| src/miss\_alignment/models/\_compact.py              |      116 |       94 |     19% |6-49, 53-58, 63-91, 95-99, 104-132, 136-140, 145-173, 177-181, 186-225, 228-232, 252-280, 288-304, 321-347, 350-364 |
| src/miss\_alignment/models/\_resnet.py               |      104 |       27 |     74% |50-68, 71-90, 180, 192, 204 |
| src/miss\_alignment/models/models.py                 |      184 |      149 |     19% |50-53, 76-122, 140-155, 159-160, 164-190, 205-294, 299-305, 320-345, 350-351, 371-377, 381, 385-398, 409-419, 423-440, 444-445, 462-467, 471-491, 501-523 |
| src/miss\_alignment/prepare\_stacks.py               |       46 |       35 |     24% |38-62, 86-103, 139-167 |
| src/miss\_alignment/preprocessing.py                 |        6 |        0 |    100% |           |
| src/miss\_alignment/train.py                         |      117 |       96 |     18% |26, 82-338 |
| src/miss\_alignment/utils.py                         |       15 |       10 |     33% |20-27, 35-36, 45-46 |
| **TOTAL**                                            | **1782** |  **878** | **51%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/warpem/miss-alignment/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/warpem/miss-alignment/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/warpem/miss-alignment/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/warpem/miss-alignment/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fwarpem%2Fmiss-alignment%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/warpem/miss-alignment/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.