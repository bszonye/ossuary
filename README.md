# oddly 🎲 probability analysis toolkit
<!-- head -->

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://hatch.pypa.io/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Types: MyPy](https://img.shields.io/badge/types-mypy-blue.svg)](https://github.com/python/mypy)

<!-- cut -->
A Python package for analyzing dice and other randomizers.

Features:

* Exact values for probability computations.
* Statistical analysis (mean, mode, standard deviation).
* Graph plotting with visualization of median and other quantiles.
* Support for symbolic as well as numeric dice.
* Command-line evaluator for ease of use.
* Python API for extensibility.

Installation
------------
```
pip install oddly
pip install matplotlib  # optional graphing support
```

Commands
--------
Roll a die.
```
oddly d6
```
Roll d4+1 three times and add the results.
```
oddly '3 @ (d4+1)'
```
Roll 4d6 and add the highest 3 dice.
```
oddly 'Dice.NdX(4, d6, kh=3).sum()'
```
Take the middle die out of three d20 rolls.
```
oddly 'Dice.NdX(3, d20, km=1).sum()'
```
Graph 12d6 with the results partitioned by deciles.
```
oddly '(12 @ d6).plot(q=10)'
```

<!-- cut -->
---
💀 _fin_
