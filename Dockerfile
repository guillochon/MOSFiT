# MOSFiT CPU runtime for the OSG / SkyPortal analysis service.
#
# A generic MOSFiT runtime: the osg-skyportal-plugin ships mosfit_wrapper.py +
# mosfit_bridge.py per-job, so no plugin code is baked in. Built from this repo
# (this fork's fetcher is local-only, so fits run fully offline on a worker).
# ZTF transmission curves are baked in so ZTF-band fits never touch the network.
#
# Default image: core deps + the mpi extra (mpi4py) for cluster/OSG. The
# sedona extra (PyTorch) is not installed; `import mosfit` must work without
# torch. Add a second stage or `--extra sedona` only if a worker needs
# sesn_sedona.
FROM python:3.14.7-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# openmpi so the optional mpi4py extra builds and imports (local fits can use
# --max-cores; cluster jobs may still want MPI). curl for the SVO filter fetch.
# git: astrocats imports GitPython, which needs a git executable at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates git \
        libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mosfit
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY mosfit ./mosfit

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/mosfit/.venv/bin:$PATH"

# Frozen lock, no dev extras, mpi for OSG/cluster. Do not install sedona/torch.
RUN uv sync --frozen --no-dev --extra mpi --no-cache

# Bake ZTF (Palomar/ZTF.{g,r,i}) transmission curves into the installed package
# so ZTF-band fits resolve filters locally instead of downloading from SVO at
# runtime (a worker has no network).
RUN FILTERS=$(python -c "import mosfit, os; print(os.path.join(os.path.dirname(mosfit.__file__), 'modules', 'observables', 'filters'))") \
    && for b in g r i; do \
        curl -fsSL "http://svo2.cab.inta-csic.es/svo/theory/fps3/fps.php?PhotCalID=Palomar/ZTF.${b}/AB" \
            -o "$FILTERS/Palomar_ZTF.${b}_AB.xml"; \
    done \
    && for b in g r i; do \
        test -s "$FILTERS/Palomar_ZTF.${b}_AB.xml" || { echo "missing baked ZTF ${b} filter"; exit 1; }; \
    done

# Fail the build if the runtime isn't importable, or if torch leaked in.
RUN python -c "\
import sys; \
import mosfit; \
from mosfit.fitter import Fitter; \
assert 'torch' not in sys.modules, 'import mosfit must not import torch'; \
assert __import__('importlib.util').util.find_spec('torch') is None, 'torch must not be installed (omit the sedona extra)'; \
print('mosfit OK (no torch)')"

ENTRYPOINT ["python"]
