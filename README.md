# FriendlyFit

[![Build Status](https://img.shields.io/travis/guillochon/FriendlyFit.svg)](https://travis-ci.org/guillochon/FriendlyFit)
[![Coverage Status](https://coveralls.io/repos/github/guillochon/FriendlyFit/badge.svg?branch=master)](https://coveralls.io/github/guillochon/FriendlyFit?branch=master)
[![Python Version](https://img.shields.io/badge/python-3.4%2C%203.5-blue.svg)](https://www.python.org)

`FriendlyFit` is a Python package that will fit semi-analytical model light curves to observed supernova data. Data is pulled automatically from the Open Supernova Catalog by name, and thus the code can be used to fit *any* supernova within that database.

To run `FriendlyFit`, simply pass the list of event names to the program via the `-e` flag:

```bash
python -m friendlyfit -e SN2015bn
```

FriendlyFit is parallelized and can be run in parallel by simply prepending `mpirun -np #`, where `#` is the number of processors in your machine +1 for the master process. So, if you computer has 4 processors, the command would be:

```bash
mpirun -np 5 python -m friendlyfit -e SN2015bn
```

A Jupyter notebook `friendlyfit.ipynb` is provided in the main directory, and by default will show output from the last `FriendlyFit` run.
