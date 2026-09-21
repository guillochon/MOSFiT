# conda-forge

This directory is a **template** for updating
[conda-forge/mosfit-feedstock](https://github.com/conda-forge/mosfit-feedstock)
when MOSFiT 2.x is published to PyPI. conda-forge still builds from the PyPI
sdist; it does not use `uv` or `uv.lock`.

MOSFiT 2 is pure Python (Numba JITs at runtime). The recipe is `noarch: python`
so one artifact works on Linux, macOS, and Windows.

## Submit / bump a version

1. **astrocats first.** conda-forge currently has `astrocats 0.3.37`. MOSFiT 2
   needs `astrocats >= 0.5.0` (already on PyPI). Open a version bump on
   [conda-forge/astrocats-feedstock](https://github.com/conda-forge/astrocats-feedstock)
   and wait for it to merge before the MOSFiT feedstock will solve.
2. Tag `v2.0.0` (or later), `uv build`, and upload the sdist/wheel to PyPI.
3. Copy `recipe/meta.yaml` into the feedstock. Set `sha256` to the PyPI sdist
   hash (`uv hash dist/mosfit-*.tar.gz` or the PyPI “Download files” page).
4. Drop the old feedstock bits that 2.0 no longer supports:
   - `setuptools` as the build backend (use **hatchling**)
   - `pytorch` and `cython` as required run dependencies
   - `numpy <=1.26.4` and `skip: true  # [py >= 313]`
   - `schwimmbad <0.4` (need `schwimmbad >=0.4.2`)
   - `mpi4py` as a hard run dependency (it is the optional `mpi` extra)
   - `test.sh`, `SN2006le.json`, `PTF10hgi.txt`, `event_list.txt`
5. Keep tests to `import mosfit`, `mosfit --version`, and `python test.py`
   (generative dummy events; no catalog download). The sdist also includes
   `mosfit/tests/LSQ12dlf.json` if the feedstock wants a real-event smoke
   (`python mosfit/tests/_test_lsq12dlf.py`).

Local check that the sdist is conda-shaped (no conda required):

```bash
uv build
python -m venv .venv-sdist
.venv-sdist/bin/pip install dist/mosfit-*.tar.gz
.venv-sdist/bin/python -c "import mosfit; print(mosfit.__version__)"
```
