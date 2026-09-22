.. _outputs:

=======
Outputs
=======

The model structure used in ``MOSFiT`` makes it amenable to producing outputs from models that need not be fit against any particular transient. In this section we walk through how the user can extract various data products.

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

    mosfit -e ./LSQ12dlf.json -m slsn --extra-instruments UVIS --extra-bands F218W --extra-systems Vega

.. _mock:

-----------------------------------------------
Mock light curves in a magnitude-limited survey
-----------------------------------------------

Generating a light curve from a model in ``MOSFiT`` is achieved by simply not passing any event to the code with the ``-e`` option. The command below will dump out a default number of parameter draws to a ``walkers.h5`` file in the ``products`` folder:

.. code-block:: bash

    mosfit -m slsn

By default, these light curves are *exact* model predictions: MOSFiT does not add survey-style scatter when no event file is supplied. Uncertainties that appear alongside magnitudes follow the ordinary likelihood / variance parameters wired into the chosen model—not a residual Gaussian-process layer.

If the user wishes to produce mock observations for a given instrument, they should use the ``-l`` option, which sets a limiting magnitude and then randomly draws observations based upon the flux error implied by that limiting magnitude (the second argument to ``-l`` sets the variance of the limiting magnitude from observation to observation). For example, if the user wishes to generate mock light curves as they might be observed by LSST assuming a limiting magnitude of 23 for all bands, they would execute:

.. code-block:: bash

    mosfit -m slsn -l 23 0.5 --extra-bands u g r i z y --extra-instruments LSST

.. _chain:

----------------
Saving the chain
----------------

Because the chain can be quite large, by default ``MOSFiT`` does not output the full chain to disk. Doing so is achieved by passing ``MOSFiT`` the ``-c`` option:

.. code-block:: bash

    mosfit -m slsn -e ./LSQ12dlf.json -c

The chain is written as compressed HDF5 (``products/chain.h5``), with datasets ``samples`` (shape ``ntemps × nwalkers × nsteps × nparams``) and ``param_names``. Load it in Python with::

    import h5py
    with h5py.File('products/chain.h5', 'r') as hf:
        samples = hf['samples'][:]
        param_names = [n.decode() for n in hf['param_names'][:]]

Note that the outputted chain includes both the burn-in and post-burn-in phases of the fitting procedure. The position of each walker in the chain as a function of time can be visualized using the included ``mosfit.ipynb`` Jupyter notebook.

Memory can be quite scarce on some systems, and storing the chain in memory can sometimes lead to out of memory errors (it is the dominant user of memory in ``MOSFiT``). This can be mitigated to some extent by automatically thinning the chain if it gets too large with the ``-M`` option, where the argument to ``-M`` is in MB. Below, we limit the chain to a gigabyte, which should be sufficient for most modern systems:

.. code-block:: bash

    mosfit -m slsn -e ./LSQ12dlf.json -M 1000

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

* ``bands``: Band names associated with each outputted epoch; the ordering in ``extras.json`` should match the ordering of other observables such as ``seds``.

* ``dense_times``: Times at which luminosity was computed (units: days). These are sampled more densely than the input observations as dense sampling is required for an accurate integration of the luminosity.

* ``dense_luminosities``: Luminosity of transient at each observation epoch (units: ergs / s).

.. _lynx:

---------------------------------------------------
Rest-frame SEDs for external light-curve simulators
---------------------------------------------------

Everything above produces *observed* photometry: ``MOSFiT`` redshifts the SED,
reddens it along the line of sight, integrates it through bandpasses and, with
``-l``, adds survey noise. Simulation frameworks such as `LightCurveLynx
<https://lightcurvelynx.readthedocs.io>`_ do all of that themselves, and ask a
source model for one thing only — rest-frame flux density as a function of
phase and wavelength. The ``--lynx`` flag reports exactly that view:

.. code-block:: bash

    mosfit -m slsn --lynx --lynx-wavelengths 1000 25000 100 -S 100 -N 1000

This path does not need the sampling or plotting stacks, and there is a
lightweight install that leaves them out: see :ref:`lightweight`.

The convention matches ``SEDModel.compute_sed``: flux density in **nJy**, as
the source would appear at **10 pc**, in the **rest frame**, with no redshift,
no time dilation and no extinction applied. Redshift, luminosity distance,
explosion time and extinction are pinned to the values that make this
well-defined, so that the calling simulator owns them rather than fighting
``MOSFiT`` over them. Bandpasses and ``-l`` play no part and are ignored; use
``-S`` to set how many phases are sampled. With no ``-D``, the ensembler is
used: the realizations are prior draws, not a posterior, and the ensembler is
the one sampler the lightweight install of :ref:`lightweight` carries.

Two files land in ``products``:

* ``lynx_seds.h5``, holding a flat ``seds`` block of shape ``(n_realization,
  n_phase, n_wave)`` together with the ``phases`` and ``wavelengths`` grids,
  the ``fractions`` (unit-cube coordinates) behind each realization and the
  ``free_parameter_names`` they correspond to. Nothing here requires knowledge
  of the catalog schema, so it can be read straight into a training pipeline::

    import h5py
    with h5py.File('products/lynx_seds.h5', 'r') as hf:
        seds = hf['seds'][:]            # nJy, (n_realization, n_phase, n_wave)
        phases = hf['phases'][:]        # days since explosion
        waves = hf['wavelengths'][:]    # Angstroms, rest frame
        fractions = hf['fractions'][:]  # (n_realization, n_free), in [0, 1]

* ``lynx_manifest.json``, describing every parameter — prior range, units, log
  flag and position in the walker vector — alongside the wavelength and phase
  ranges over which the model is defined.

.. _lynx-api:

^^^^^^^^^^^^^^^^^^^^^^^
Calling this in-process
^^^^^^^^^^^^^^^^^^^^^^^

A wrapper that evaluates one sample at a time should not pay for file output
and console traffic on every call. :class:`mosfit.lynx.LynxSource` is the same
machinery without them::

    import numpy as np
    from mosfit.lynx import LynxSource

    source = LynxSource(
        model='slsn',
        phases=np.linspace(0.0, 200.0, 100),
        wavelengths=np.linspace(1000.0, 25000.0, 100))

    source.free_parameter_names()
    source.parameter_manifest()
    source.minwave(), source.maxwave(), source.minphase(), source.maxphase()

    sed = source.compute_sed(parameters={'mejecta': 5.0, 'vejecta': 1.0e4})

``compute_sed`` returns an ``(n_phase, n_wave)`` array in nJy. Parameters may
be given as physical values, or as unit-cube ``fractions`` if the caller is
doing its own prior sampling; unspecified free parameters take the midpoint of
their prior. Changing the phase or wavelength grid rebuilds the model, which is
expensive, so hold one ``LynxSource`` per grid and vary only the parameters.
