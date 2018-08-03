.. _outputs:

=======
Outputs
=======

The model structure used in ``MOSFiT`` makes it ammenable to producing outputs from models that need not be fit against any particular transient. In this section we walk through how the user can extract various data products.

.. _light-curve-options:

-------------------
Light curve options
-------------------

By default, ``MOSFiT`` will only compute model observations at the times a particular transient was observed using the instrument for which it was observed at those times. If a transient is sparsely sampled, this will likely result in a choppy light curve with no prediction for intra-observation magnitudes/fluxes.

.. _smooth:

Smooth light curves
===================

A smooth output light curve can be produced using the ``-S`` option, which when passed no argument returns the light curve with *every* instrument's predicted observation at all times. If given an argument (e.g. ``-S 100``), ``MOSFiT`` will return every instrument's predicted observation at all times *plus* an additional :math:`S` observations between the first and last observation.

.. _extrapolated:

Extrapolated light curves
=========================

If the user wishes to extrapolate beyond the first and last observations, the ``-E`` option will extend the predicted observations by :math:`E` days both before and after the first/last detections.

.. _unobserved:

Predicted observations that were not observed
=============================================

The user may wish to generate light curves for a transient in instruments/bands for which the transient was not observed; this can be accomplished using the ``--extra-bands``, ``extra-instruments``, ``extra-bandsets``, and ``extra-systems`` options. For instance, to generate LCs in Hubble's UVIS filter F218W in the Vega system in addition to the observed bands, the user would enter:

.. code-block:: bash

    mosfit -e LSQ12dlf -m slsn --extra-instruments UVIS --extra-bands F218W --extra-systems Vega

.. _mock:

-----------------------------------------------
Mock light curves in a magnitude-limited survey
-----------------------------------------------

Generating a light curve from a model in ``MOSFiT`` is achieved by simply not passing any event to the code with the ``-e`` option. The command below will dump out a default number of parameter draws to a ``walkers.json`` file in the ``products`` folder:

.. code-block:: bash

    mosfit -m slsn

By default, these light curves will be the *exact* model predictions, they will not account for any observational error. If Gaussian Processes were used (by default they are enabled for all models), the output predictions will include an ``e_magnitude`` value that is set by the variance predicted by the GP model; if not, the ``variance`` parameter from maximum likelihood is used.

If the user wishes to produce mock observations for a given instrument, they should use the ``-l`` option, which sets a limiting magnitude and then randomly draws observations based upon the flux error implied by that limiting magnitude (the second argument to ``-l`` sets the variance of the limiting magnitude from observation to observation). For example, if the user wishes to generate mock light curves as they might be observed by LSST assuming a limiting magnitude of 23 for all bands, they would execute:

.. code-block:: bash

    mosfit -m slsn -l 23 0.5 --extra-bands u g r i z y --extra-instruments LSST

.. _chain:

----------------
Saving the chain
----------------

Because the chain can be quite large (a full chain for a model with 15 free parameters, 100 walkers, and 20000 iterations will occupy ~120 MB of disk space), by default ``MOSFiT`` does not output the full chain to disk. Doing so is achieved by passing ``MOSFiT`` the ``-c`` option:

.. code-block:: bash

    mosfit -m slsn -e LSQ12dlf -c

Note that the outputted chain includes both the burn-in and post-burn-in phases of the fitting procedure. The position of each walker in the chain as a function of time can be visualized using the included ``mosfit.ipynb`` Jupyter notebook.

Memory can be quite scarce on some systems, and storing the chain in memory can sometimes lead to out of memory errors (it is the dominant user of memory in ``MOSFiT``). This can be mitigated to some extent by automatically thinning the chain if it gets too large with the ``-M`` option, where the argument to ``-M`` is in MB. Below, we limit the chain to a gigabyte, which should be sufficient for most modern systems:

.. code-block:: bash

    mosfit -m slsn -e LSQ12dlf -M 1000

.. _arbitrary:

-----------------
Arbitrary outputs
-----------------

Internally, ``MOSFiT`` is storing the outputs of each module in a single dictionary that is handed down through the execution tree like a hot potato. This dictionary behaves like a list of global variables, and when a model is executed from start to finish, it will be filled with values that were produced by all modules included in that module.

The user can dump any of these variables to a supplementary file ``extras.json`` by using the ``-x`` option, followed by the name of the variable of interest. For instance, if the user is interested in the spectral energy distributions and bolometric luminosities associated with the SLSN model of a transient, they can simply pass the ``seds`` and ``dense_luminosities`` keys to ``-x``:

.. code-block:: bash

    mosfit -m slsn -x seds dense_luminosities

Below is an inexhaustive list of keys available; a full list of keys can be displayed by adding the ``-x`` option with no arguments.

* ``seds``: Spectral energy distributions at each observation epoch over each photometric filter requested (units: ergs / s / Angstrom). To obtain a broadband SED, one should add the ``'white'`` filter to the ``MOSFiT`` command via ``--band-list white``.

# ``bands``: Band names associated with each outputted epoch, the ordering in ``extras.json`` should match the ordering of other observables such as ``seds``.

* ``dense_times``: Times at which luminosity was computed (units: days). These are sampled more densely than the input observations as dense sampling is required for an accurate integration of the luminosity.

* ``dense_luminosities``: Luminosity of transient at each observation epoch (units: ergs / s).
