Getting Started
===============
.. toctree::

Once installed, ``MOSFiT`` can be run from any directory, and it's
typically convenient to make a new directory for your project.

.. code:: bash

    mkdir mosfit_runs
    cd mosfit_runs

Then, to run ``MOSFiT``, pass an event name to the program via the
``-e`` flag (the default model is a simple Nickel-Cobalt decay with
diffusion):

.. code:: bash

    python -m mosfit -e LSQ12dlf

Different models (several are distributed with ``MOSFiT``) can be fit to
supernovae using the model flag ``-m``:

.. code:: bash

    python -m mosfit -e LSQ12dlf -m slsn

Multiple events can be fit in succession by passing a list of names
separated by spaces (names containing spaces can be specified using
quotation marks):

.. code:: bash

    python -m mosfit -e LSQ12dlf SN2015bn "SDSS-II SN 5751"

``MOSFiT`` is parallelized and can be run in parallel by prepending
``mpirun -np #``, where ``#`` is the number of processors in your
machine +1 for the master process. So, if you computer has 4 processors,
the above command would be:

.. code:: bash

    mpirun -np 5 python -m mosfit -e LSQ12dlf

``MOSFiT`` can also be run without specifying an event, which will yield
a collection of light curves for the specified model described by the
priors on the possible combinations of input parameters specified in the
``parameters.json`` file. This is useful for determining the range of
possible outcomes for a given theoretical model:

.. code:: bash

    mpirun -np 5 python -m mosfit -i 0 -m magnetar

The code outputs JSON files for each event/model combination that each
contain a set of walkers that have been relaxed into an equilibrium
about the combinations of parameters with the maximum likelihood. This
output is visualized via an example Jupyter notebook (``mosfit.ipynb``)
included with the software in the main directory, which by default shows
output from the last ``MOSFiT`` run.

To upload fits back to the open astronomy catalogs, users simply pass
the ``-u`` to the the code:

.. code:: bash

    python -m mosfit -e LSQ12dlf -m slsn -u

After running ``MOSFiT`` to completion, and if the fits satisfy some
quality checks, the model fits will be displayed on the open catalogs
within 48 hours of their submission.
