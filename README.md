<p align="center"><img src="logo.png" align="right" alt="MOSFiT" width="300"/></p>
#MOSFiT
[![Build Status](https://img.shields.io/travis/guillochon/MOSFiT.svg)](https://travis-ci.org/guillochon/MOSFiT)
[![Coverage Status](https://coveralls.io/repos/github/guillochon/MOSFiT/badge.svg?branch=master)](https://coveralls.io/github/guillochon/MOSFiT?branch=master)
[![Python Version](https://img.shields.io/badge/python-3.4%2C%203.5-blue.svg)](https://www.python.org)

`MOSFiT` (Module Open-Source Fitter for Transients) is a Python package that will fit semi-analytical model light curves to observed transient data. Data is currently pulled automatically from the Open Supernova Catalog by name, and thus the code can be used to fit *any* supernova within that database, or any database that shares that format (such as the [Open TDE Catalog](https://tde.space) or the [Open Nova Catalog](https://opennova.space))<br clear="all">

To run `MOSFiT`, simply pass the list of event names to the program via the `-e` flag:

```bash
python -m mosfit -e SN2015bn
```

MOSFiT is parallelized and can be run in parallel by simply prepending `mpirun -np #`, where `#` is the number of processors in your machine +1 for the master process. So, if you computer has 4 processors, the command would be:

```bash
mpirun -np 5 python -m mosfit -e SN2015bn
```

A Jupyter notebook `mosfit.ipynb` is provided in the main directory, and by default will show output from the last `MOSFiT` run.
