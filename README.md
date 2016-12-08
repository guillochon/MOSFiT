<p align="center"><img src="logo.png" align="left" alt="MOSFiT" width="300"/></p>
[![Build Status](https://img.shields.io/travis/guillochon/MOSFiT.svg)](https://travis-ci.org/guillochon/MOSFiT)
[![Coverage Status](https://coveralls.io/repos/github/guillochon/MOSFiT/badge.svg?branch=master)](https://coveralls.io/github/guillochon/MOSFiT?branch=master)
[![Python Version](https://img.shields.io/badge/python-3.4%2C%203.5%2C%203.6-blue.svg)](https://www.python.org)
[![PyPI version](https://badge.fury.io/py/mosfit.svg)](https://badge.fury.io/py/mosfit)

`MOSFiT` (**M**oduluar **O**pen-**S**ource **Fi**tter for **T**ransients) is a Python 3.x package that performs maximum likelihood analysis to fit semi-analytical model predictions to observed transient data. Data can be provided by the user, or can be pulled automatically from the [Open Supernova Catalog](https://sne.space) by its name, and thus the code can be used to fit *any* supernova within that database, or any database that shares the format described in the [OSC schema](https://github.com/astrocatalogs/supernovae/blob/master/SCHEMA.md) (such as the [Open TDE Catalog](https://tde.space) or the [Open Nova Catalog](https://opennova.space)).<br clear="all">

##Getting Started

To install `MOSFiT` into your Python environment, clone the package and then run the `setup.py` file:

```bash
git clone https://github.com/guillochon/MOSFiT.git
cd MOSFiT
python setup.py install
```

Once installed, MOSFiT can be run from any directory, and it's typically convenient to make a new directory for your project.

```bash
mkdir mosfit_runs
cd mosfit_runs
```

Then, to run `MOSFiT`, pass an event name to the program via the `-e` flag (the default model is a simple Nickel-Cobalt decay with diffusion):

```bash
python -m mosfit -e LSQ12dlf
```

Multiple events can be fit in succession by passing a list of names separated by spaces (names containing spaces can be specified using quotation marks):

```bash
python -m mosfit -e LSQ12dlf LSQ12dlf "SDSS-II SN 5751"
```

MOSFiT is parallelized and can be run in parallel by prepending `mpirun -np #`, where `#` is the number of processors in your machine +1 for the master process. So, if you computer has 4 processors, the above command would be:

```bash
mpirun -np 5 python -m mosfit -e LSQ12dlf
```

MOSFiT can also be run without specifying an event, which will yield a collection of light curves for the specified model described by the priors on the possible combinations of input parameters specified in the `parameters.json` file. This is useful for determining the range of possible outcomes for a given theoretical model:

```bash
mpirun -np 5 python -m mosfit -i 0 -m magnetar
```

The code outputs JSON files for each event/model combination that each contain a set of walkers that have been relaxed into an equilibrium about the combinations of parameters with the maximum likelihood. This output is visualized via an example Jupyter notebook (`mosfit.ipynb`) included with the software in the main directory, which by default shows output from the last `MOSFiT` run.
