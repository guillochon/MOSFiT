<p align="center"><img src="logo.png" align="left" alt="MOSFiT" width="300"/></p>
[![Build Status](https://img.shields.io/travis/guillochon/MOSFiT.svg)](https://travis-ci.org/guillochon/MOSFiT)
[![Coverage Status](https://coveralls.io/repos/github/guillochon/MOSFiT/badge.svg?branch=master)](https://coveralls.io/github/guillochon/MOSFiT?branch=master)
[![Python Version](https://img.shields.io/badge/python-2.7%2C%203.4%2C%203.5%2C%203.6-blue.svg)](https://www.python.org)
[![PyPI version](https://badge.fury.io/py/mosfit.svg)](https://badge.fury.io/py/mosfit)
[![Documentation Status](https://readthedocs.org/projects/mosfit/badge/?version=latest)](http://mosfit.readthedocs.io/en/latest/?badge=latest)

`MOSFiT` (**M**odular **O**pen-**S**ource **Fi**tter for **T**ransients) is a Python 2.7/3.x package that performs maximum likelihood analysis to fit semi-analytical model predictions to observed transient data. Data can be provided by the user, or can be pulled automatically from the [Open Supernova Catalog](https://sne.space) by its name, and thus the code can be used to fit *any* supernova within that database, or any database that shares the format described in the [OSC schema](https://github.com/astrocatalogs/supernovae/blob/master/SCHEMA.md) (such as the [Open TDE Catalog](https://tde.space) or the [Open Nova Catalog](https://opennova.space)). With the use of an optional upload flag, fits performed by users can then be uploaded back to the aforementioned catalogs.<br clear="all">

##Installation

`MOSFiT` is available on pip, and can be installed in the standard way:

```bash
pip install mosfit
```

To assist in the development of `MOSFiT`, the repository should be cloned from GitHub and then installed into your Python environment via the `setup.py` file:

```bash
git clone https://github.com/guillochon/MOSFiT.git
cd MOSFiT
python setup.py install
```

##Using MOSFiT

For detailed instructions on using MOSFiT, please see our documentation on RTD: http://mosfit.readthedocs.io/
