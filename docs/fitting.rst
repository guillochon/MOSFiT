.. _fitting:

============
Fitting data
============

The primary purpose of ``MOSFiT`` is to fit models of transients to observed data. In this section we cover "how" of fitting, and how the user should interpret their results.

.. _public:

-----------
Event data
-----------

``MOSFiT`` does **not** download event JSON from the Open Astronomy Catalogs (or similar services). You must pass **paths to files** with the ``-e`` flag: catalog-format JSON (see schema links below), or ASCII tables that the built-in converter can turn into JSON.

.. code-block:: bash

    mosfit -m slsn -e ./my_supernova.json

If ``-e`` does not resolve to an existing file, ``MOSFiT`` exits with an error explaining that network fetch by transient name is no longer supported.

.. _private:

----------------------
ASCII / catalog JSON
----------------------

If you already have catalog-format JSON, pass it directly. The format follows the Open Catalog schema (historical examples: `Supernova SCHEMA <https://github.com/astrocatalogs/supernovae/blob/master/SCHEMA.md>`_).

For ASCII tables, pass the path to the file(s); ``MOSFiT`` converts them where needed:

.. code-block:: bash

    mosfit -e path/to/my/ascii/file/my_transient.dat
    mosfit -e path/to/my/folder/of/ascii/files/*.dat

If the ASCII file matches required columns closely, conversion may proceed quietly; otherwise you will be prompted to map columns. When multiple files share one format, questions are driven from the first file.

.. _sampling:

----------------
Sampling Options
----------------

``MOSFiT`` offers three ways to sample the parameter space. Nested sampling
with ``dynesty`` (>= 3.1) is the **default**. Ensemble MCMC (``emcee``) and
UltraNest remain available.

Samplers are selected via ``-D`` / ``--method``:

- ``-D dynesty`` — dynamic nested sampling (default)
- ``-D ensembler`` — affine-invariant ensemble MCMC
- ``-D ultranest`` — reactive nested sampling (``ultranest`` is imported at
  runtime and is not a required dependency)

Likelihoods can be evaluated in parallel with ``--max-cores N`` (process pool;
works on Windows). ``mpirun`` still takes precedence when MPI is used.

.. _dynesty:
.. _nester:

Nested sampling with dynesty
=============================

In ``MOSFiT``, nested sampling via the ``dynesty`` package uses *dynamic*
nested sampling (`full documentation <http://dynesty.rtfd.io>`_).

Whereas ensemble-based approaches can only estimate the information content of their posteriors via heuristic information metrics such as the WAIC (see :ref:`scoring`), nested sampling directly evaluates the evidence for a given model, and provides a (statistical) estimate of its error. Nested sampling also yields many more useful samples of the posterior for the purposes of visualizing its structure; it is not uncommon for a run to provide tens of thousands of informative samples, as compared to ensemble-based approach that may only yield a few hundred.

However, nested sampling is a much more complicated algorithm than ensemble-based MCMC and thus is potentially prone to failures that can be difficult to track down. Additionally, ``dynesty`` currently does not offer the ability to restart if the sampling is prematurely terminated; thus, it is advisable to use nested sampling in conjunction with the ``-R`` flag, which when used with ``-D dynesty`` sets the termination criterion based upon the expected remaining evidence gain (default ``dlogz`` is ``0.02``).

The nested sampler is the default, and can also be selected explicitly via ``-D dynesty``. Fracking (scipy minimization during burn-in) applies to the ensemble sampler; it is not used on the dynesty path. ``--num-walkers`` / ``-N`` does **not** set dynesty's number of live points; MOSFiT uses ``nlive = 20 * ndim``.

The status line **Efficiency** is dynesty's sampling efficiency
(``100 × nested iterations / likelihood calls``), not CPU use and not
fraction of the fit completed.

.. _baselining-batching:

Baselining and batching
-----------------------

When performing a nested sampling run, the user might notice that there are two phases to the process: "baselining" and "batching". In the baselining phase, ``dynesty`` samples from the posterior repeatedly to obtain the log of the evidence :math:`\log Z` (the N-dimensional volume integral of the posterior), for which it estimates the remaining :math:`\Delta \log Z`. Once that remaining evidence is smaller than the threshold set with ``-R``, baselining ceases and batching begins.

In batching, ``dynesty`` fleshes out the posterior such that even regions of lower probability that may not be dominating the evidence integral are resolved with high fidelity. This process continues until a stopping criterion is met. Batching often takes longer than baselining.

CPU use may drop between bursts of likelihood evaluations: dynesty rebuilds
its bounding ellipsoids on the parent process (workers idle), and after each
batch MOSFiT merges runs and checks the stopping function. That is expected
and does not mean the job has hung.

.. _ensembler:

Ensemble-based MCMC
===================

In ensemble-based Markov chain Monte Carlo, a collection of parameter positions (called "walkers") are evolved in the parameter space according to simple rules based upon the positions of their neighbors. This approach is simple, flexible, and is able to deal with several pathologies in posteriors that can cause issues in other samplers. In ``MOSFiT`` we implement this sampling using the parallel-tempered sampler available within the ``emcee`` package, although a single temperature is used by default (note that the parallel-tempered sampler is now deprecated as of ``emcee`` version ``3.0``, and ``MOSFiT`` will eventually deprecate this option as well).

While ``MOSFiT`` also performs minimization during the burn-in phase to find the global minima within the posterior, it should be noted that ``emcee`` on its own has been found to have poor convergence to the posterior for problems with greater than about 10 dimensions (`Huijser et al. 2015 <https://arxiv.org/abs/1509.02230>`_). As many models provided with ``MOSFiT`` have a dimension similar to this number, care should be taken when using this sampler to ensure that convergence has been achieved.

The ensemble-based MCMC can be selected via the ``-D`` flag: ``-D ensembler`` (it is no longer the default).

.. _initialization:

Initialization
--------------

When initializing, walkers are drawn randomly from the prior distributions of all free parameters, unless the ``-w`` option was passed to initialize from a previous run (see :ref:`previous`). By default, any drawn walker that has a defined, non-infinite score will be retained, unless the ``-d`` option is used, which by default only draws walkers above the average walker score drawn so far, or the numeric value specified by the user (warning: this option can often make the initial drawing phase last a *long* time).

.. _restricting:

Restricting the data used
-------------------------

By default, ``MOSFiT`` will attempt to use all available data when fitting a model. If the user wishes, they can exclude specific instruments from the fit using the ``--exclude-instruments`` option, specific photometric bands using the ``--exclude-bands`` option, specific sources of data (e.g. papers or surveys) using ``--exclude-sources``, and particular wave bands via ``--exclude-kinds``. The source is specified using the source ID number, visible on the Open Astronomy Catalog page for each transient as well as in the input file. For example

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn --exclude-sources 2

will exclude all data tagged with source ID ``2`` in your input JSON.

To exclude times from a fit, the user can specify a range of MJDs that will be included using the ``-L`` option, e.g.:

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn -L 55000 56000

will limit the data fitted for LSQ12dlf to lie between MJD 55000 and MJD 56000.

Finally, ``--exclude-kinds`` can be used to exclude particular wave bands (e.g. radio, X-ray, infrared) from the fitting process. By default, models will not fit against data that is not specified as being supported via a ``'supports'`` attribute in the model JSON file, but this can be overridden by setting ``--exclude-kinds none``.

As an example, assuming a user wants to fit the ``ic`` model to a transient that happens to have radio data, but would like to exclude the radio data from that fit, they would run the following command:

.. code-block:: bash

    mosfit -e ./SN2004gk.json -m ic --exclude-kinds radio

.. _number:

Number of walkers
-----------------

The ensemble sampler used in ``MOSFiT`` is a variant of ``emcee``'s multi-temperature sampler ``PTSampler``. Pass a number of temperatures with ``-T`` and walkers per temperature with ``-N``. If one temperature is used (the default), the total number of walkers is whatever is passed to ``-N``, otherwise it is :math:`N*T`. These flags apply to ``-D ensembler``, not to dynesty live points.

.. _duration:

Duration of fitting
-------------------

The duration of an **ensemble** (``-D ensembler``) run is set with the ``-i`` option, unless the ``-R`` or ``-U`` options are used (see :ref:`convergence <convergence>`). Generally, unless the model has only a few free parameters or was initialized very close to the solution of highest-likelihood, the user should not expect good results unless ``-i`` is set to a few thousand or more.

For the default ``dynesty`` sampler, prefer ``-R`` (remaining evidence / ``dlogz``, default ``0.02``) rather than a large ``-i``. ``-i``, ``-b``/``-p``, and ``-f`` describe ensemble burn-in and walking, not nested sampling.

.. _burning:

Burning in a model
------------------

Burn-in and fracking apply to ``-D ensembler``. Nested sampling does not use this path (``--no-fracking`` is implicit for ``dynesty``).

Unless the solution for a given dataset is known in advance, the initial period of searching for the true posterior distribution involves finding the locations of the solutions of highest likelihood. In ``MOSFiT``, various ``scipy`` routines are employed in an alternating fashion with a Gibbs-like affine-invariant ensemble evolution, which we have found more robustly locates the true global likelihood minimas. The period of alternation between optimization (called "fracking" in ``MOSFiT``) and sampling (called "walking" in ``MOSFiT``) is controlled by the ``-f`` option, with the total burn-in duration being controlled by the ``-b``/``-p`` options. If ``-b``/``-p`` are not set, the burn-in is set to run for half the total number of iterations specified by ``-i``.

As an example, the following will run the burn-in phase for 2000 iterations, the post burn-in for 3000 iterations more (for a total of 5000), fracking every 100th iteration:

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn -f 100 -i 5000 -b 2000

All :ref:`convergence <convergence>` metrics are computed *after* the burn-in phase, as the operations employed during burn-in do *not* preserve detailed balance. During burn-in, the solutions of highest likelihood are over-represented, and thus the posteriors should not be trusted until the :ref:`convergence <convergence>` criteria are met beyond the burn-in phase.

.. _ultranest:

Nested sampling with ultranest
==============================

For complicated posteriors with multiple modes or for problems of high dimension (ten dimensions or greater), nested sampling is often a superior choice versus ensemble-based methods.
In ``MOSFiT``, reactive nested sampling is also available via the ``ultranest`` package. More information about ``ultranest`` can be found at https://johannesbuchner.github.io/UltraNest/.

Select it with ``-D ultranest``. ``ultranest`` is imported only when that sampler is chosen; it is not installed by ``uv sync`` unless you add it yourself.

Ultranest supports resuming from a previous run if you set the output path (``-o myoutputdirectory``).

If you have ``mpi4py`` installed, Ultranest supports running with MPI (``mpiexec -np 8 mosfit``).

.. _switching:

Switching between samplers
==========================

After completing a nested sampling run, it is often useful to draw parameter combinations from the large collection of samples generated to perform additional analysis (particularly for data-intensive tasks, such as analyzing a collection of model SEDs). This can be easily done by loading the output from the previous run with the ``ensembler`` method (via ``-D ensembler``), and setting ``MOSFiT`` to run in generative mode with ``-G``,

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn -w products/walkers.h5 -G -N 100

where above we specify that we would like 100 parameter combinations from the ``dynesty`` output. The weights determined with ``dynesty`` will be used to proportionately draw walkers for ``ensembler``, yielding a sample that properly maps to the posterior determined by the nested sampling. As the above does not perform any additional sampling, the user does not need to specify an event to compare against, and can simply omit the ``-e`` flag and its argument(s).

Because ``dynesty`` currently does not support restarts, the opposite situation of using ``ensembler`` outputs to initialize ``dynesty`` is not possible.

.. _io:

--------------------------
Input and output locations
--------------------------

The paths of the various inputs and outputs are set by a few different options in ``MOSFiT``. The first time ``MOSFiT`` runs in a directory, it will make local copies of the ``models`` and ``jupyter`` folders distributed with the code (unless ``--no-copy-at-launch`` option is passed), and will *not* copy the files again unless they are deleted or the user passes the ``--force-copy-at-launch`` option.

By default, ``MOSFiT`` searches the local ``models`` folder copied to the run directory to find model JSON and their corresponding parameter JSON files to use for runs. If the user wishes to use custom parameter files for their runs instead, they can specify the paths to these files using the ``-P`` option.

``MOSFiT`` outputs are always written to a local ``products`` directory. Without ``--quick-save``, canonical names are ``walkers.h5``, ``extras.json`` (when ``-x`` is used), and ``chain.h5`` (when ``-c`` is used)—no duplicate per-event filenames. With ``--quick-save``, filenames are prefixed with the transient being fit so multiple runs stay distinct (e.g. ``LSQ12dlf_walkers.h5``).

The ``-s`` option appends a suffix to ``--quick-save`` output filenames, e.g.:

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn --quick-save -s mysuffix

includes ``mysuffix`` in those names (e.g. ``LSQ12dlf_walkers_mysuffix.h5``).

``walkers.h5`` stores the same merged event+model catalog payload historically written as ``walkers.json`` (gzip-compressed JSON bytes under ``entry_json``). Pass it to ``-w`` to seed a later run; legacy ``.json`` walker files are still accepted.

The MCMC chain, when requested with ``-c``, is written once as HDF5 (``chain.h5``) unless ``--quick-save`` is set, where it is ``<event>_chain[_suffix].h5`` instead. The file contains a ``samples`` dataset with axes ``(temperature, walker, step, parameter)`` and a ``param_names`` dataset listing the free parameters.

.. _fixing:

-----------------------
Fixing model parameters
-----------------------

Individual parameters can be locked to fixed values with the ``-F`` option, which will either assume the default specified in the model JSON file (if no value is provided):

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn -F kappa

Or, will assume the value specified by the user:

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn -F mejecta 3.0

Multiple fixed variables can be specified by chaining them together, with any user-prescribed variables following the variable names:

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn -F kappa mejecta 3.0

If you have a prior for a given variable (not a single value), it is best to modify your local ``parameters.json`` file. For instance, to place a Gaussian prior on ``vejecta`` in the SLSN model, replace the default ``parameters.json`` snippet, which looks like this:

.. code-block:: json

    "vejecta":{
        "min_value":5.0e3,
        "max_value":2.0e4
    },

with the following:

.. code-block:: json

    "vejecta":{
        "class":"gaussian",
        "mu":1.0e4,
        "sigma":0.5e3,
        "min_value":1.0e3,
        "max_value":1.0e5
    },

Flat, log flat, gaussian, and power-law priors are available in ``MOSFiT``; see the `parameters_test.json <https://github.com/guillochon/MOSFiT/blob/master/mosfit/models/default/parameters_test.json>`_ file in the ``default`` model for examples on how to set each prior type.



.. _other-prior:

Other prior
=======================


If you have another prior following a function not specified above, you can create your own prior by using the ``arbitrary`` class prior. To start with, you need to create a file (e.g., ``filename.csv``) which storing the information of your function:

.. code-block:: txt

    X   Y
    0   1.1
    1   1.2
    2   1.3
    .   .
    .   .
    .   .
    


with X as the parameter value axis and Y as the PDF.

Save the file where you will run ``MOSFiT``, and edit the ``parameters.json`` as follows:

.. code-block:: json

    "vejecta":{
        "class":"Arbitrary",
        "filename":"filename.csv",
        "min_value":1.0e3,
        "max_value":1.0e5
    },

.. _previous:

-------------------------------
Initializing from previous runs
-------------------------------

The user can use the ensemble parameters from a prior ``MOSFiT`` run to draw their initial conditions for a new run using the ``-w`` option. Assuming that ``LSQ12dlf-mysuffix.json`` contains results from a previous run, the user can draw walker positions from it by passing it to the ``-w`` option:

.. code-block:: bash

    mosfit -e ./LSQ12dlf.json -m slsn -w products/walkers.h5

If the file contains more walkers than requested by the new run, walker positions will be drawn verbatim from the input file, otherwise walker positions will be "jittered" by a small amount so no two walkers share identical parameters.

Note that while the outputs of nested sampling runs can be initialized *from*, they cannot themselves be initialized from previous runs, as the nested sampling approach must sample from the full prior volume.
