.. _outputs:

Outputs
=======

The model structure used in ``MOSFiT`` makes it ammenable to producing outputs from models that need not be fit against any particular transient. In this section we walk through how the user can extract various data products.

Light Curve Predictions
-----------------------

.. _light-curve

Generating a light curve from a model in ``MOSFiT`` is achieved by simply not passing any event to the code with the ``-e`` flag. The command below will dump out a default number of parameter draws to a ``walkers.json`` file in the ``products`` folder::

    mosfit -m slsn

By default, these light curves will be the *exact* model predictions, they will not account for any observational error. If Gaussian Processes were used (by default they are enabled for all models), the output predictions will include an ``e_magnitude`` value that is set by the variance predicted by the GP model; if not, the ``variance`` parameter from maximum likelihood is used. If the user wishes to produce mock observations for a given instrument, they should use the ``-l`` flag, which sets a limiting magnitude and then randomly draws observations based upon the flux error implied by that limiting magnitude (the second argument to ``-l`` sets the variance of the limiting magnitude from observation to observation). For example, if the user wishes to generate mock light curves as they might be observed by LSST, they would execute::

    mosfit -m slsn -l 23 0.5 --extra-bands u g r i z y --extra-instruments LSST

Saving the Chain
----------------

.. _chain

Because the chain can be quite large (a full chain for a model with 15 free parameters, 100 walkers, and 20000 iterations will occupy ~120 MB of disk space), by default ``MOSFiT`` does not output the full chain to disk. Doing so is achieved by passing ``MOSFiT`` the ``-c`` flag::

    mosfit -m slsn -e LSQ12dlf -c

Note that the outputted chain includes both the burn-in and post-burn-in phases of the fitting procedure. The position of each walker in the chain as a function of time can be visualized using the included ``mosfit.ipynb`` Jupyter notebook.

Memory can be quite scarce on some systems, and storing the chain in memory can sometimes lead to out of memory errors (it is the dominant user of memory in ``MOSFiT``). This can be mitigated to some extent by automatically thinning the chain if it gets too large with the ``-M`` flag, where the argument to ``-M`` is in MB. Below, we limit the chain to a gigabyte, which should be sufficient for most modern systems::

    mosfit -m slsn -e LSQ12dlf -M 1000

Arbitrary Outputs
-----------------

.. _arbitrary

Internally, ``MOSFiT`` is storing the outputs of each module in a single dictionary that is handed down through the execution tree like a hot potato. This dictionary behaves like a list of global variables, and when a model is executed from start to finish, it will be filled with values that were produced by all modules included in that module.

The user can dump any of these variables to a supplementary file ``extras.json`` by using the ``-x`` flag, followed by the name of the variable of interest. For instance, if the user is interested in the bolometric luminosity of the SLSN model, they can simply pass the ``seds`` and ``dense_luminosities`` keys to ``-x``::

    mosfit -m slsn -x seds dense_luminosities
