.. _fitting:

============
Fitting data
============

The primary purpose of ``MOSFiT`` is to fit models of transients to observed data. In this section we cover "how" of fitting, and how the user should interpret their results.

.. _public:

-----------
Public data
-----------

``MOSFiT`` is deeply connected to the Open Catalogs (The Open Supernova Catalog, the Open Tidal Disruption Catalog, etc.), and the user can directly fit their model against any data provided by those catalogs. The Open Catalogs store names for each transient, and the user can access any transient by any known name of that transient. As an example, both of the commands below will fit the same transient:

.. code-block:: bash

    mosfit -m slsn -e PTF11dij
    mosfit -m slsn -e CSS110406:135058+261642

While the Open Catalogs do their best to maintain the integrity of the data they contain, there is always the possibility that the data contains errors, so users are encouraged to spot check the data they download before using it for any scientific purpose. A common error is that the data has been tagged with the wrong photometric system, or has not been tagged with a photometric system at all and uses a different system from what is commonly used for a given telescope/instrument/band. Users are encouraged to immediately report any issues with the public data on the GitHub issues page assocated with that catalog (e.g. the Open Supernova Catalog's `issue page <https://github.com/astrocatalogs/supernovae/issues>`_).

.. _private:

------------
Private data
------------

If you have private data you would like to fit, the most robust way to load the data into ``MOSFiT`` is to directly construct a JSON file from your data that conforms to the `Open Catalog Schema <https://github.com/astrocatalogs/supernovae/blob/master/SCHEMA.md>`_. This way, the user can specify all the data that ``MOSFiT`` can use for every single observation in a precise way. All data provided by the Open Catalogs is provided in this form, and if the user open up a typical JSON file downloaded from one of these catalogs, they will find that each observation is tagged with all the information necessary to model it.

Of course, it is more likely that the data a user will have handy will be in another form, typically an ASCII table where each row presents a single (or multiple) observations. ``MOSFiT`` includes a conversion feature where the user can simply pass the path to the file(s) to convert:

.. code-block:: bash

    mosfit -e path/to/my/ascii/file/my_transient.dat
    mosfit -e path/to/my/folder/of/ascii/files/*.dat

In some cases, if the ASCII file is in a simple form with columns that match all the required columns, ``MOSFiT`` will silently convert the input files into JSON files, a copy of which will be saved to the current run directory. In most cases however, the user will be prompted to answer a series of questions about the data in a "choose your own adventure" style. If passed a list of files, ``MOSFiT`` will assume all the files share the same format and the user will only be asked questions about the first file.

If the user so chooses, they may *optionally* upload their data directly to the Open Catalogs with the ``-u`` option. This will make their observational data publicly accessible on the Open Catalogs:

.. code-block:: bash

    mosfit -e path/to/my/ascii/file/my_transient.dat -u

Note that this step is completely optional, users do not have to share their data publicly to use ``MOSFiT``, however it is the fastest way for your data to appear on the Open Catalogs. If a user believes they have uploaded any private data in error, they are encouraged to immediately contact the :ref:`maintainers <maintainers>`.

.. _sampling:

----------------
Sampling Options
----------------

``MOSFiT`` at present offers two ways to sample the parameter space: An ensemble-based MCMC (implemented with the ```emcee`` package), and a nested sampling approach (implement with the ``dynesty`` package). The ensemble-based approach is presently the default sampler used in ``MOSFiT``, although the nested sampler is likely to replace it as the default in a future version.

Samplers are selected via the ``-D`` option: ``-D ensembler`` for the ensemble-based approach, ``-D nester`` for the nested sampling approach. The two approaches are described below.

.. _ensembler:

Ensemble-based MCMC
===================

In ensemble-based Markov chain Monte Carlo, a collection of parameter positions (called "walkers") are evolved in the parameter space according to simple rules based upon the positions of their neighbors. This approach is simple, flexible, and is able to deal with several pathologies in posteriors that can cause issues in other samplers. In ``MOSFiT`` we implement this sampling using the parallel-tempered sampler available within the ``emcee`` package, although a single temperature is used by default (note that the parallel-tempered sampler is now deprecated as of ``emcee`` version ``3.0``, and ``MOSFiT`` will eventually deprecate this option as well).

While ``MOSFiT`` also performs minimization during the burn-in phase to find the global minima within the posterior, it should be noted that ``emcee`` on its own has been found to have poor convergence to the posterior for problems with greater than about 10 dimensions (`Huijser et al. 2016 <https://arxiv.org/abs/1509.02230>`_). As many models provided with ``MOSFiT`` have a dimension similar to this number, care should be taken when using this sampler to ensure that convergence has been achieved.

.. _initialization:

Initialization
--------------

When initializing, walkers are drawn randomly from the prior distributions of all free parameters, unless the ``-w`` option was passed to initialize from a previous run (see previous_). By default, any drawn walker that has a defined, non-infinite score will be retained, unless the ``-d`` option is used, which by default only draws walkers above the average walker score drawn so far, or the numeric value specified by the user (warning: this option can often make the initial drawing phase last a *long* time).

.. _restricting:

Restricting the data used
-------------------------

By default, ``MOSFiT`` will attempt to use all available data when fitting a model. If the user wishes, they can exclude specific instruments from the fit using the ``--exclude-instruments`` option, specific photometric bands using the ``--exclude-bands`` option, and specific sources of data (e.g. papers or surveys) using ``--exclude-sources``. The source is specified using the source ID number, visible on the Open Astronomy Catalog page for each transient as well as in the input file. For example

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn --exclude-sources 2

will exclude all data from the paper that has the source ID number 2 on the Open Astronomy Catalog page.

To exclude times from a fit, the user can specify a range of MJDs that will be included using the ``-L`` option, e.g.:

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn -L 55000 56000

will limit the data fitted for LSQ12dlf to lie between MJD 55000 and MJD 56000.

.. _number:

Number of walkers
-----------------

The sampler used in ``MOSFiT`` is a variant of ``emcee``'s multi-temperature sampler ``PTSampler``, and thus the user can pass both a number of temperatures to use with ``-T`` in addition to the number of walkers ``-N`` per temperature. If one temperature is used (the default), the total number of walkers is simply whatever is passed to ``-N``, otherwise it is :math:`N*T`.

.. _duration:

Duration of fitting
-------------------

The duration of the ``MOSFiT`` run is set with the ``-i`` option, unless the ``-R`` or ``-U`` options are used (see :ref:`convergence <convergence>`). Generally, unless the model has only a few free parameters or was initialized very close to the solution of highest-likelihood, the user should not expect good results unless ``-i`` is set to a few thousand or more.

.. _burning:

Burning in a model
------------------

Unless the solution for a given dataset is known in advance, the initial period of searching for the true posterior distribution involves finding the locations of the solutions of highest likelihood. In ``MOSFiT``, various ``scipy`` routines are employed in an alernating fashion with a Gibbs-like affine-invariant ensemble evolution, which we have found more robustly locates the true global likelihood minimas. The period of alternation between optimization (called "fracking" in ``MOSFiT``) and sampling (called "walking" in ``MOSFiT``) is controlled by the ``-f`` option, with the total burn-in duration being controlled by the ``-b``/``-p`` options. If ``-b``/``-p`` are not set, the burn-in is set to run for half the total number of iterations specified by ``-i``.

As an example, the following will run the burn-in phase for 2000 iterations, the post burn-in for 3000 iterations more (for a total of 5000), fracking every 100th iteration:

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn -f 100 -i 5000 -b 2000

All :ref:`convergence <convergence>` metrics are computed *after* the burn-in phase, as the operations employed during burn-in do *not* preserve detailed balance. During burn-in, the solutions of highest likelihood are over-represented, and thus the posteriors should not be trusted until the :ref:`convergence <convergence>` criteria are met beyond the burn-in phase.

.. _nester:

Nested sampling
===============

.. _io:

--------------------------
Input and output locations
--------------------------

The paths of the various inputs and outputs are set by a few different options in ``MOSFiT``. The first time ``MOSFiT`` runs in a directory, it will make local copies of the ``models`` and ``jupyter`` folders distributed with the code (unless ``--no-copy-at-launch`` option is passed), and will *not* copy the files again unless they are deleted or the user passes the ``--force-copy-at-launch`` option.

By default, ``MOSFiT`` searches the local ``models`` folder copied to the run directory to find model JSON and their corresponding parameter JSON files to use for runs. If the user wishes to use custom parameter files for their runs instead, they can specify the paths to these files using the ``-P`` option.

``MOSFiT`` outputs are always written to a local ``products`` directory, with the default filename being set to the name of the transient being fit (e.g. ``LSQ12dlf.json`` for LSQ12dlf). The user can append a suffix to the output filename using the ``-s`` option, e.g.:

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn -s mysuffix

will write to the file ``LSQ12dlf-mysuffix.json``. A copy of the output will also always be dumped to ``walkers.json`` in the same directory. The same suffix will applied to any additional outputs requested by the user, such as the ``chain.json`` and ``extras.json`` files.

.. _fixing:

-----------------------
Fixing model parameters
-----------------------

Individual parameters can be locked to fixed values with the ``-F`` option, which will either assume the default specified in the model JSON file (if no value is provided):

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn -F kappa

Or, will assume the value specified by the user:

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn -F mejecta 3.0

Multiple fixed variables can be specified by chaining them together, with any user-prescribed variables following the variable names:

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn -F kappa mejecta 3.0

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

.. _previous:

-------------------------------
Initializing from previous runs
-------------------------------

The user can use the ensemble parameters from a prior ``MOSFiT`` run to draw their initial conditions for a new run using the ``-w`` option. Assuming that ``LSQ12dlf-mysuffix.json`` contains results from a previous run, the user can draw walker positions from it by passing it to the ``-w`` option:

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn -w LSQ12dlf-suffix.json

If the file contains more walkers than requested by the new run, walker positions will be drawn verbatim from the input file, otherwise walker positions will be "jittered" by a small amount so no two walkers share identical parameters.
