.. _installation:

============
Installation
============

Several installation methods for ``MOSFiT`` are outlined below. If you run into
issues, `open a new issue <https://github.com/guillochon/mosfit/issues>`_ on
GitHub.

``MOSFiT`` requires **Python 3.11 or newer** (3.11–3.14 are tested). Runtime
dependencies are declared in ``pyproject.toml`` (including ``astrocats>=0.5.0``,
``dynesty>=3.1``, NumPy 2, and Astropy 7). There is no ``setup.py``.

.. _source:

----------------------
Installing from source
----------------------

**Platforms:** macOS, Linux, and Windows

For development work, clone the repository and sync the project with
`uv <https://docs.astral.sh/uv/>`__:

.. code-block:: bash

    git clone https://github.com/guillochon/MOSFiT.git
    cd MOSFiT
    uv sync

This creates a local virtual environment in ``.venv``. Run MOSFiT with
``uv run mosfit ...``, or activate the environment and call ``mosfit``
directly. A ``.python-version`` file pins a modern CPython for local ``uv``
use; any interpreter satisfying ``requires-python = ">=3.11"`` is allowed.

Optional extras:

.. code-block:: bash

    uv sync --extra mpi      # mpi4py (needs a system MPI library)
    uv sync --extra sedona   # PyTorch, only for the SESN SEDONA emulator
    uv sync --extra docs     # Sphinx stack for building these pages

Combine extras as needed, e.g. ``uv sync --extra mpi --extra sedona``.

Equivalent editable pip install from the same ``pyproject.toml``:

.. code-block:: bash

    pip install -e .
    pip install -e ".[mpi,sedona,docs]"

.. _pip:

-------------------
Installing with pip
-------------------

**Platforms:** macOS, Linux, and Windows

Installing ``MOSFiT`` with pip from PyPI:

.. code-block:: bash

    pip install mosfit
    pip install 'mosfit[sedona]'   # SESN SEDONA emulator
    pip install 'mosfit[mpi]'      # mpi4py

Published wheels may lag this development branch.

.. _anaconda:

-------------------------------
Setting up MOSFiT with Anaconda
-------------------------------

**Platforms:** macOS, Linux, and Windows

The `Anaconda <https://www.anaconda.com/download>`__ or Miniconda
distribution is a convenient Python environment. After installing conda,
a released ``MOSFiT`` can be installed via:

.. code-block:: bash

    conda install -c conda-forge mosfit

MOSFiT **2.0** is a hatchling package (no ``setup.py``). After this release is
on PyPI, conda-forge needs a feedstock bump: use the template in ``recipe/``
of this repository. **astrocats >= 0.5.0** must be on conda-forge first (the
channel still has 0.3.37 at the time of writing). PyTorch and mpi4py are
optional extras, not required conda run dependencies. Until that feedstock
PR lands, install 2.0 from source with ``uv sync`` or from PyPI with pip.


.. _lightweight:

---------------------------------------
Lightweight install for rest-frame SEDs
---------------------------------------

If all you need is the rest-frame SED path (:ref:`lynx`) — driving ``MOSFiT``
as a source model from an external light-curve simulator, or generating
training data — you do not need the sampling stack. Nothing on that path
imports ``emcee``, ``dynesty`` or ``numba``, and a CI job keeps it that way.

``pyproject.toml`` declares a ``lynx`` dependency group holding just that
subset. Because extras can only ever *add* packages, the lightweight install
works by suppressing ``MOSFiT``'s own dependencies and then installing the
group on its own:

.. code-block:: bash

    uv venv
    uv pip install --no-deps -e .     # or: --no-deps mosfit
    uv pip install --group lynx

This drops ``dynesty``, ``numba`` and ``llvmlite`` — the last two being by far
the largest of the omitted wheels. ``emcee`` is kept, since the command-line
``--lynx`` run draws from the priors with the ensembler; ``mosfit.lynx.
LynxSource`` does not need it.

Two caveats worth stating plainly:

* ``astrocats`` imports ``matplotlib``, ``seaborn`` and ``pandas`` at module
  scope, so those arrive regardless of what ``MOSFiT`` asks for. They are not
  something this install path can avoid.
* The ``tde`` model JIT-compiles its viscous transform, so it additionally
  needs ``numba``. Every other bundled model runs without it.

Anything beyond the SED path — fitting real events, plotting, nested sampling
— wants the full install described above.

.. _docker:

------
Docker
------

A CPU runtime image is built from the repository ``Dockerfile``. It installs
MOSFiT with ``uv sync --frozen --extra mpi`` on Python 3.14.7 and **does not**
include PyTorch. ``import mosfit`` works without the ``sedona`` extra. Local
fits inside the image can use ``--max-cores``; MPI is available when jobs are
launched with ``mpirun``. Build with Docker from the repository root; see
``.github/workflows/docker-publish.yml`` for the published image tags.
