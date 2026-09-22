# Changelog

All notable changes to MOSFiT are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## How to cut a release

1. Move everything under **Unreleased** into a new `## [X.Y.Z] - YYYY-MM-DD`
   section (use the UTC date of the tag).
2. Leave an empty **Unreleased** section at the top for the next cycle.
3. Set `__version__` in `mosfit/__init__.py` to `X.Y.Z` if it is not already.
4. Tag `vX.Y.Z` and push the tag. The `Publish` workflow builds the sdist and
   wheel and uploads them to PyPI via trusted publishing; check its run before
   announcing the release.
5. Open a version bump on
   [conda-forge/mosfit-feedstock](https://github.com/conda-forge/mosfit-feedstock)
   using `recipe/meta.yaml` as the starting point (see `recipe/README.md`).

## [Unreleased]

### Added

- `--lynx`, which reports the model as a rest-frame SED rather than as observed
  photometry: flux density in nJy at 10 pc on a wavelength grid of the user's
  choosing (`--lynx-wavelengths MIN MAX N`), with no redshift, time dilation or
  extinction applied. This is the shape external light-curve simulators such as
  [LightCurveLynx](https://lightcurvelynx.readthedocs.io) expect from a source
  model. Redshift, luminosity distance, explosion time and extinction are pinned
  so that the caller owns them. Writes `products/lynx_seds.h5` (a flat
  `(realization, phase, wavelength)` block plus the grids and the unit-cube
  coordinates behind each realization) and `products/lynx_manifest.json`.
- `mosfit.lynx.LynxSource`, the in-process form of the same thing, for wrappers
  that call MOSFiT per sample rather than through the CLI. `compute_sed(times,
  wavelengths, parameters)` returns an `(n_phase, n_wave)` array in nJy.
- `Model.parameter_manifest()`, describing every parameter's prior range, units,
  log flag and position in the walker vector, so an external sampler can map its
  own draws onto MOSFiT's unit cube.
- `Model.minwave()` / `maxwave()` / `minphase()` / `maxphase()`, reporting where
  a model is actually defined so a caller can decide when to extrapolate.
- A `lynx` dependency group in `pyproject.toml` holding only what the
  rest-frame SED path needs, for installs that never fit or plot:
  `uv pip install --no-deps -e . && uv pip install --group lynx`. Drops
  `dynesty`, `numba` and `llvmlite`. A CI job installs exactly that set and
  runs `--lynx` against it, so an eager import of the fitting stack fails
  there rather than in a user's environment.

### Changed

- MOSFiT's submodules are now imported on first access rather than eagerly by
  `mosfit/__init__.py`, and `fitter` imports the samplers at the point it
  chooses one. `import mosfit.lynx` therefore costs neither the plotting nor
  the sampling stack. `mosfit.plotting` and friends still resolve as before.
- The `Viscous` transform's numba kernels moved to a private module imported
  when the transform runs, so `numba` is no longer imported by every MOSFiT
  session. Module package scans now skip `_`-prefixed files.
- `--lynx` defaults to the ensembler when no sampler is given: rest-frame SEDs
  are prior draws, with no likelihood to nest against, and the ensembler is the
  one sampler the lightweight `lynx` group installs. An explicit `-D` wins.

## [2.0.1] - 2026-09-21

Bug fixes for two paths 2.0.0 left broken. No API or packaging changes.

### Fixed

- Generative runs (`mosfit -m <model>` with no event, i.e. `-i 0`) crashed
  under the default `dynesty` sampler with `AttributeError: 'Nester' object
  has no attribute '_results'`: with no likelihood to nest against,
  `Nester.run()` returned without leaving results behind. The nester now
  draws from the priors as the ensembler does, so generative mode works
  without `-D ensembler`. Those draws are equally weighted and are written
  one realization apiece rather than being resampled against their weights,
  which had been duplicating some and dropping others.
- The `--limiting-magnitude` mock-survey noise model mixed boolean-masked
  and full-length arrays, so it raised `ValueError` as soon as a model
  produced a bolometric luminosity or a radio flux density alongside
  magnitudes. Rows are now masked consistently. A draw at or below zero
  flux becomes an upper limit rather than a `NaN` that silently dropped the
  epoch from the mock light curve.

### Changed

- The PyPI version badge now comes from shields.io. The badge.fury
  endpoint it used had gone stale and was still advertising 1.3 after
  2.0.0 was published.
- Documentation: the built-in model table is a `list-table` rather than a
  hand-aligned grid table, whose column rules had drifted out of
  alignment and stopped parsing; `autosectionlabel_prefix_document` is
  on, so section labels no longer collide with explicit `.. _target:`
  names; and Read the Docs installs graphviz for the inheritance
  diagrams. The docs now build clean, and CI builds them with `-W`.

## [2.0.0] - 2026-09-21

First 2.x release: `uv`/`pyproject.toml` packaging, Python 3.11-3.14, a dynesty
default sampler, and a NumPy 2 / Astropy 7 runtime.

### Added

- `uv` / `pyproject.toml` packaging (hatchling). Python **3.11–3.14**.
- Optional extras: `mpi` (mpi4py), `sedona` (PyTorch, SESN SEDONA only), `docs`.
- `--max-cores N` local process pool for likelihoods (Windows spawn-safe).
- GitHub Actions test workflow (`uv run python mosfit/tests/run_all.py`) with
  line coverage fail-under 70%.
- Catalog fixtures `mosfit/tests/PS1-10jh.json` and `mosfit/tests/LSQ12dlf.json`,
  plus an SLSN likelihood test on LSQ12dlf.
- A simulated event per built-in model under `mosfit/tests/events`, generated by
  `mosfit/tests/make_event_fixtures.py`, and `_test_transient_types.py`, which
  loads and briefly fits every built-in transient type against them.
- `_test_csv_input.py`, covering the ASCII/CSV input path: it converts
  `mosfit/tests/events/sim_default.csv`, checks the conversion reproduces the
  matching JSON fixture, and fits the result.
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

[Unreleased]: https://github.com/guillochon/MOSFiT/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/guillochon/MOSFiT/compare/v1.3...v2.0.0
[1.3]: https://github.com/guillochon/MOSFiT/releases/tag/v1.3
