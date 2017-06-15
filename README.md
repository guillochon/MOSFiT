<p align="center"><img src="logo.png" align="left" alt="MOSFiT" width="300"/></p>
<a href="https://travis-ci.org/guillochon/MOSFiT"><img src="https://img.shields.io/travis/guillochon/MOSFiT.svg" alt="Build Status"></a>
<a href="https://coveralls.io/github/guillochon/MOSFiT?branch=master"><img src="https://coveralls.io/repos/github/guillochon/MOSFiT/badge.svg?branch=master" alt="Coverage Status"></a>
<a href="https://www.python.org"><img src="https://img.shields.io/badge/python-2.7%2C%203.4%2C%203.5%2C%203.6-blue.svg" alt="Python Version"></a>
<a href="https://badge.fury.io/py/mosfit"><img src="https://badge.fury.io/py/mosfit.svg" alt="PyPI version"></a>
<a href="http://mosfit.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mosfit/badge/?version=latest" alt="Documentation Status"></a>

`MOSFiT` (**M**odular **O**pen-**S**ource **Fi**tter for **T**ransients) is a Python 2.7/3.x package that performs maximum likelihood analysis to fit semi-analytical model predictions to observed transient data. Data can be provided by the user, or can be pulled automatically from the [Open Supernova Catalog](https://sne.space) by its name, and thus the code can be used to fit *any* supernova within that database, or any database that shares the format described in the [OSC schema](https://github.com/astrocatalogs/supernovae/blob/master/SCHEMA.md) (such as the [Open TDE Catalog](https://tde.space) or the [Open Nova Catalog](https://opennova.space)). With the use of an optional upload flag, fits performed by users can then be uploaded back to the aforementioned catalogs.<br clear="all">

## Installation

`MOSFiT` is available on `conda` and `pip`, and can be installed using:

```bash
conda install -c conda-forge mosfit
```

or:

```bash
pip install mosfit
```

For a development install of `MOSFiT`, the repository should be cloned from GitHub and then installed into your Python environment via the `setup.py` script:

```bash
git clone https://github.com/guillochon/MOSFiT.git
cd MOSFiT
python setup.py develop
```

## Using MOSFiT

For detailed instructions on using MOSFiT, please see our documentation on RTD: <http://mosfit.readthedocs.io/>
