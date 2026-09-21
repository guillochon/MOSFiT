<p align="center"><img src="logo.png" align="left" alt="MOSFiT" width="300"/></p>
<a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python Version"></a>
<a href="https://badge.fury.io/py/mosfit"><img src="https://badge.fury.io/py/mosfit.svg" alt="PyPI version"></a>
<a href="https://mosfit.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mosfit/badge/?version=latest" alt="Documentation Status"></a>
<a href="https://ascl.net/1710.006"><img src="https://img.shields.io/badge/ascl-1710.006-blue.svg?colorB=262255" alt="ascl:1710.006" /></a>

`MOSFiT` (**M**odular **O**pen-**S**ource **Fi**tter for **T**ransients) is a Python 3.11+ package for fitting, sharing, and estimating the parameters of transients via user-contributed transient models. This is **MOSFiT 2.0**. Data for a transient can be provided by the user in a wide range of formats (JSON, ASCII tables, CDS, LaTeX).<br clear="all">

See [CHANGELOG.md](CHANGELOG.md) for unreleased work and versioned notes.

## Installation

Development installs use [uv](https://docs.astral.sh/uv/) and Python 3.11+:

```bash
git clone https://github.com/guillochon/MOSFiT.git
cd MOSFiT
uv sync
```

Then run MOSFiT with `uv run mosfit ...`, or activate `.venv`.

Published installs are also available via `conda` and `pip`. conda-forge will
ship 2.0 after a feedstock update (see `recipe/`); `astrocats >= 0.5.0` must
land on conda-forge first:

```bash
conda install -c conda-forge mosfit
```

or:

```bash
pip install mosfit
```

MPI support (`mpi4py`) is optional and needs a system MPI library:

```bash
uv sync --extra mpi
```

The SESN SEDONA emulator (`sesn_sedona`) needs PyTorch. Install that extra with:

```bash
uv sync --extra sedona
```

or, for a published install, `pip install 'mosfit[sedona]'`. Default models do not require it.


## Using MOSFiT

The default sampler is nested sampling with dynesty. Pass local catalog JSON (or ASCII) with `-e`. Use `--max-cores` for local process-pool parallelism (no MPI required):

```bash
mosfit -e ./my_transient.json -m slsn --max-cores 10
mosfit -e mosfit/tests/PS1-10jh.json -m tde --max-cores 10 -R
```

Switch to ensemble MCMC with `-D ensembler`, or UltraNest with `-D ultranest`. Fits write products under `products/` (including `walkers.h5`, and optionally `chain.h5` if run with `-c`). For detailed instructions, see the documentation on RTD: <https://mosfit.readthedocs.io/>
