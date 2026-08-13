# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/warpem/miss-alignment/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                 |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/miss\_alignment/\_\_init\_\_.py                  |       12 |        2 |     83% |       7-8 |
| src/miss\_alignment/\_\_main\_\_.py                  |        2 |        2 |      0% |       1-3 |
| src/miss\_alignment/\_cli.py                         |       10 |        1 |     90% |         9 |
| src/miss\_alignment/alignment/\_\_init\_\_.py        |        3 |        0 |    100% |           |
| src/miss\_alignment/alignment/correlation.py         |       32 |       32 |      0% |     1-112 |
| src/miss\_alignment/alignment/optimize\_global.py    |      184 |      120 |     35% |82-93, 103-124, 164-171, 177-190, 202-209, 215-228, 292-334, 356-365, 371-444, 450-517 |
| src/miss\_alignment/alignment/optimize\_iterative.py |       83 |       13 |     84% |67, 111, 116, 130, 168, 172-175, 179, 230-244 |
| src/miss\_alignment/alignment/optimize\_spline.py    |      116 |       88 |     24% |126-158, 181-318, 366-402 |
| src/miss\_alignment/alignment/parallel.py            |        5 |        2 |     60% |     28-30 |
| src/miss\_alignment/alignment/statistics.py          |        9 |        0 |    100% |           |
| src/miss\_alignment/alignment/tilt\_series.py        |       55 |       18 |     67% |   174-269 |
| src/miss\_alignment/alignment/utils.py               |        7 |        4 |     43% |     28-32 |
| src/miss\_alignment/data/\_\_init\_\_.py             |        2 |        0 |    100% |           |
| src/miss\_alignment/data/\_augmentation.py           |       39 |        0 |    100% |           |
| src/miss\_alignment/data/\_reconstruction\_worker.py |      147 |        0 |    100% |           |
| src/miss\_alignment/data/io.py                       |      119 |       42 |     65% |45, 49, 53, 58, 78, 81, 91-93, 96, 99-103, 107, 112, 211-270 |
| src/miss\_alignment/data/shift\_generation.py        |      124 |        6 |     95% |148, 291-294, 297 |
| src/miss\_alignment/data/training\_datamodule.py     |      102 |       77 |     25% |41-55, 126-168, 172-173, 177-178, 193-258, 267, 286-310 |
| src/miss\_alignment/data/training\_dataset.py        |       32 |        0 |    100% |           |
| src/miss\_alignment/distributed/\_\_init\_\_.py      |        6 |        0 |    100% |           |
| src/miss\_alignment/distributed/config.py            |       30 |        1 |     97% |        55 |
| src/miss\_alignment/distributed/manager.py           |       81 |       46 |     43% |44, 48-85, 112-114, 117-126, 129-131 |
| src/miss\_alignment/distributed/provisioner.py       |      208 |       42 |     80% |26-52, 73, 86-87, 90, 121-122, 181, 245, 284-285, 288, 330-331, 377, 380-384 |
| src/miss\_alignment/distributed/queue.py             |      105 |        6 |     94% |148-150, 196, 201, 204, 207 |
| src/miss\_alignment/distributed/worker.py            |       91 |       13 |     86% |41-49, 64-65, 76, 92, 139 |
| src/miss\_alignment/infer.py                         |       77 |       11 |     86% |98-100, 109-121 |
| src/miss\_alignment/models/\_\_init\_\_.py           |        4 |        0 |    100% |           |
| src/miss\_alignment/models/\_compact.py              |      116 |       94 |     19% |6-49, 53-58, 63-91, 95-99, 104-132, 136-140, 145-173, 177-181, 186-225, 228-232, 252-280, 288-304, 321-347, 350-364 |
| src/miss\_alignment/models/\_resnet.py               |      104 |       27 |     74% |50-68, 71-90, 180, 192, 204 |
| src/miss\_alignment/models/models.py                 |      181 |      146 |     19% |50-53, 76-113, 131-146, 150-151, 155-182, 198-300, 305-311, 326-351, 356-357, 377-383, 387, 391-404, 415-425, 429-446, 450-451, 468-473, 477-497, 507-529 |
| src/miss\_alignment/prepare\_stacks.py               |       36 |       28 |     22% |36-60, 84-100, 138-167 |
| src/miss\_alignment/preprocessing.py                 |        4 |        0 |    100% |           |
| src/miss\_alignment/train.py                         |      156 |      108 |     31% |40-42, 65-70, 139-268, 333-522 |
| src/miss\_alignment/utils.py                         |       36 |        3 |     92% |48, 91, 95 |
| **TOTAL**                                            | **2318** |  **932** | **60%** |           |


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