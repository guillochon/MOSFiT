# Changelog

All notable changes to MOSFiT are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## How to cut a release

1. Move everything under **Unreleased** into a new `## [X.Y.Z] - YYYY-MM-DD`
   section (use the UTC date of the tag).
2. Leave an empty **Unreleased** section at the top for the next cycle.
3. Set `__version__` in `mosfit/__init__.py` to `X.Y.Z` if it is not already.
4. Tag `vX.Y.Z`, publish the sdist/wheel to PyPI, then open a version bump on
   [conda-forge/mosfit-feedstock](https://github.com/conda-forge/mosfit-feedstock)
   using `recipe/meta.yaml` as the starting point (see `recipe/README.md`).

## [Unreleased]

Changes that have landed since the last Git tag. The work below is the MOSFiT
**2.0.0** release; move this section to `[2.0.0]` and date it when `v2.0.0` is
tagged.

### Added

- `uv` / `pyproject.toml` packaging (hatchling). Python **3.11–3.14**.
- Optional extras: `mpi` (mpi4py), `sedona` (PyTorch, SESN SEDONA only), `docs`.
- `--max-cores N` local process pool for likelihoods (Windows spawn-safe).
- GitHub Actions test workflow (`uv run python mosfit/tests/run_all.py`) with
  line coverage fail-under 70%.
- Catalog fixtures `mosfit/tests/PS1-10jh.json` and `mosfit/tests/LSQ12dlf.json`,
  plus an SLSN likelihood test on LSQ12dlf.
- This changelog. Conda-forge feedstock notes and a 2.0 recipe template under
  `recipe/`.

### Changed

- **Default sampler is dynesty** (`-D dynesty`). Pass `-D ensembler` for the
  previous ensemble MCMC default. Requires **dynesty >= 3.1**.
- Runtime floor **astrocats >= 0.5.0** from PyPI (NumPy 2.3 / Astropy 7.1).
- `import mosfit` does not import torch. SED modules lazy-load.
- Photometry caches filter interpolations on `sample_wavelengths`.
- SEDs stay rectangular `(n_obs, n_wav)` float64 arrays through photometry.
- TDE `Fallback` engine is NumPy-ized; viscous delay uses a Numba piecewise-linear
  exponential recurrence (same integral as the old interpolant).
- Docker image: `python:3.14.7-slim`, `uv sync --frozen --extra mpi`, no torch.

### Removed

- `setup.py` / `setup.cfg` / `MANIFEST.in` as the source of truth.
- Vendored astrocats 0.3.37 wheel.
- Catalog-by-name download and fit upload. Event data is local files only.
- Obsolete CLI `test.sh` and the extra fixtures it needed (`SN2006le.json`,
  `PTF10hgi.txt`, `event_list.txt`). Conda-forge tests should use `test.py` and
  `mosfit/tests/run_all.py` (see `recipe/README.md`).

### Fixed

- Generative dummy times no longer pass an empty string into `linspace` (NumPy 2).
- Nested sampling no longer rebuilds `sampler.results` every dynesty step.

## [1.3]

Last 1.x release on PyPI and conda-forge (`1.3`). Python 2-era packaging,
ensemble MCMC default, NumPy 1.x, and a required PyTorch/mpi4py conda payload.

[Unreleased]: https://github.com/guillochon/MOSFiT/compare/v1.3...HEAD
[1.3]: https://github.com/guillochon/MOSFiT/releases/tag/v1.3
