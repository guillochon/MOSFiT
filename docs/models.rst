.. _models:

======
Models
======

``MOSFiT`` has been designed to be modular and easily modifiable by users to alter, combine, and create physical models to approximate the behavior of observed transients. On this page we walk through some basics on how a user might alter an existing model shipped with the code, and how they could go about creating their own model.

.. _model-list:

-----------------------
List of built-in models
-----------------------

+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| Model name   | Description                           | Reference(s)                                                                  |
+==============+=======================================+===============================================================================+
| ``default``  | Nickel-cobalt decay                   | 1994ApJS...92..527N                                                           |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``csm``      | Interacting CSM-SNe                   | 2013ApJ...773...76C, 2017ApJ...849...70V, 2020RNAAS...4...16J                 |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``csmni``    | CSM + NiCo decay                      | See ``default`` & ``csm``                                                     |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``exppow``   | Analytical engine                     |                                                                               |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``ia``       | NiCo decay + I-band                   |                                                                               |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``ic``       | NiCo decay + radio                    |                                                                               |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``magnetar`` | Magnetar engine w/ simple SED         | `2017ApJ...850...55N <http://adsabs.harvard.edu/abs/2017ApJ...850...55N>`_    |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``magni``    | Above + NiCo decay                    |                                                                               |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``rprocess`` | Kilonova                              | `2017ApJ...851L..21V <http://adsabs.harvard.edu/abs/2017ApJ...851L..21V>`_    |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``kilonova`` | Kilonova                              | `2017ApJ...851L..21V <http://adsabs.harvard.edu/abs/2017ApJ...851L..21V>`_    |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``bns``      | Kilonova + binary params + angle      | `2021arXiv210202229N <https://ui.adsabs.harvard.edu/abs/2021arXiv210202229N>`_|
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``slsn``     | Magnetar + modified SED + constraints | `2017ApJ...850...55N <http://adsabs.harvard.edu/abs/2017ApJ...850...55N>`_    |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+
| ``tde``      | Tidal disruption events               | `2018arXiv180108221M <http://adsabs.harvard.edu/abs/2018arXiv180108221M>`_    |
+--------------+---------------------------------------+-------------------------------------------------------------------------------+

.. [*] In development.

.. _altering:

--------------------------
Altering an existing model
--------------------------

For many simple alterations to a model, such as adjusting input priors, setting variables remain free and which should be fixed, and adding/removing modules to the call stack, the user only needs to modify copies of the model JSON files. For these sorts of minor changes, no Python code should need to be modified by the user!

For this example, we'll presume that the user will be modifying the ``slsn`` model. First, the user should create a new run directory and run ``MOSFiT`` there once to copy the model JSON files to that run directory:

.. code-block:: bash

    mkdir slsn_run
    cd slsn_run
    python -m mosfit -m slsn

After running, the user will notice four directories will have been created in the run directory: ``models``, ``modules``, ``jupyter``, and ``products``. ``models`` will contain a clone of the ``models`` directory structure, with the ``parameters.json`` files for each model copied into each model folder, ``modules`` contains a clone of the ``modules`` directory structure, etc.

.. _priors:

Changing parameter priors
=========================

From your run directory, navigate into the ``models/slsn`` directory and edit the ``parameters.json`` file in your favorite text editor:

.. code-block:: bash

    cd models/slsn
    vim parameters.json

You'll notice that ``parameters.json`` file is fairly bare-bones, containing only a list of model parameters and their allowed value ranges:

.. code-block:: json

    {
        "nhhost":{
            "min_value":1.0e16,
            "max_value":1.0e23,
            "log":true
        },
        "Pspin":{
            "min_value":1.0,
            "max_value":10.0
        },
        "Bfield":{
            "min_value":0.1,
            "max_value":10.0
        },
        "Mns":{
            "min_value":1.0,
            "max_value":2.0
        },
    }

Now, change the range of allowed neutron star masses to something else:

.. code-block:: json

    {
        "Mns":{
            "min_value":1.5,
            "max_value":2.5
        },
    }

**Congratulations!** You have just modified your first MOSFiT model. It should be noted that even this very minor change, which affects the range of a single parameter, would generate a completely different model hash than the default model, distinguishing it from any other models that might have been uploaded by other users using the default settings.

You can also use more complex priors within the same file. For example:

.. code-block:: json

    {
    "Mns":{
        "class":"gaussian",
        "mu":1.4,
        "sigma":0.4,
        "min_value":0.1,
        "max_value":3.0,
        "log":false
    }
    }

A list of available priors is below; for all prior types, ``min_value`` and ``max_value`` specify the minimum and maximum allowed parameter values, and ``log`` will apply the prior to the log transform of the parameter.

+---------------+---------------------------------------------------------------+--------------------------------------------------+
| Prior name    | Equation                                                      | Additional parameters                            |
+===============+===============================================================+==================================================+
| ``parameter`` | :math:`\Pi\sim {\rm constant}`                                |                                                  |
+---------------+---------------------------------------------------------------+--------------------------------------------------+
| ``gaussian``  | :math:`\Pi\sim \exp\left(\frac{-(x-\mu)^2}{2\sigma^2}\right)` | :math:`\mu` (``mu``), :math:`\sigma` (``sigma``) |
+---------------+---------------------------------------------------------------+--------------------------------------------------+
| ``powerlaw``  | :math:`\Pi\sim x^{-\alpha}`                                   | :math:`\alpha` (``alpha``)                       |
+---------------+---------------------------------------------------------------+--------------------------------------------------+

.. _swapping:

Swapping modules
================

Let's say you want to modify the SLSN model such that transform applied to the input engine luminosity is not diffusion, but instead viscosity (if the light of a SLSN was say filtered through an accretion disk rather than a dense envelope). To make this change, the user would want to swap out the ``diffusion`` module used by ``slsn`` for the ``viscous`` module. This can be accomplished by editing the ``slsn.json`` model file. The model files are not copied into the model directories by default (as they may change from version to version of ``MOSFiT``), but a ``README`` file with the full path to the model is copied to all model folders to make it easy for the user to copy the relevant JSON files:

.. code-block:: bash

    cd models/slsn
    cp $(head -1 README)/* .
    vim slsn.json

To swap ``diffusion`` for ``viscous``, the user would remove the blocks of JSON that refer to the ``diffusion`` module:

.. code-block:: json

    {
        "kappagamma":{
            "kind":"parameter",
            "value":10.0,
            "class":"parameter",
            "latex":"\\kappa_\\gamma\\,({\\rm cm}^{2}\\,{\\rm g}^{-1})"
        },
        "diffusion":{
            "kind":"transform",
            "inputs":[
                "magnetar",
                "kappa",
                "kappagamma",
                "mejecta",
                "texplosion",
                "vejecta"
            ]
        },
        "temperature_floor":{
            "kind":"photosphere",
            "inputs":[
                "texplosion",
                "diffusion",
                "temperature"
            ]
        },
        "slsn_constraints":{
            "kind":"constraint",
            "inputs":[
                "mejecta",
                "vejecta",
                "kappa",
                "tnebular_min",
                "Pspin",
                "Mns",
                "diffusion",
                "texplosion",
                "redshift",
                "alltimes",
                "neutrino_energy"
            ]
        },
    }

and replace them with blocks appropriate for ``viscous``:

.. code-block:: json

    {
        "Tviscous":{
            "kind":"parameter",
            "value":1.0,
            "class":"parameter",
            "latex":"T_{\\rm viscous}"
        },
        "viscous":{
            "kind":"transform",
            "inputs":[
                "magnetar",
                "texplosion",
                "Tviscous"
            ]
        },
        "temperature_floor":{
            "kind":"photosphere",
            "inputs":[
                "texplosion",
                "viscous",
                "temperature"
            ]
        },
        "slsn_constraints":{
            "kind":"constraint",
            "inputs":[
                "mejecta",
                "vejecta",
                "kappa",
                "tnebular_min",
                "Pspin",
                "Mns",
                "viscous",
                "texplosion",
                "redshift",
                "alltimes",
                "neutrino_energy"
            ]
        },
    }

As can be seen above, this involved removal of definitions of free parameters that only applied to ``diffusion`` (``kappagamma``), the addition of a new free parameter for ``viscous`` (``Tviscous``), and replacement of various ``inputs`` that depended on ``diffusion`` with ``viscous``.

The user should also modify the ``parameters.json`` file to remove free parameters that are no longer in use:

.. code-block:: json

    {
        "kappagamma":{
            "min_value":0.1,
            "max_value":1.0e4,
            "log":true
        },
    }

and to the define the priors of their new free parameters:

.. code-block:: json

    {
        "Tviscous":{
            "min_value":1.0e-3,
            "max_value":1.0e5,
            "log":true
        },
    }

.. _creating:

--------------------
Creating a new model
--------------------
If users would like to create a brand new model for the ``MOSFiT`` platform, it is easiest to duplicate one of the existing models that most closely resembles
the model they wish to create.

If you go this route, we highly recommend that you `fork MOSFiT <https://github.com/guillochon/MOSFiT#fork-destination-box>`_ on GitHub and clone your fork, with development being done in the cloned ``mosfit`` directory:

.. code-block:: bash

    git clone https://github.com/your_github_username/MOSFiT.git
    cd mosfit

Copy one of the existing models as a starting point:

.. code-block:: bash

    cp -R models/slsn models/my_model_that_explains_everything


Inside this directory are two files: a ``model_name.json`` file and a ``parameters.json`` file. We must edit both files to run our new model.

First, the ``model_name.json`` file should be edited to include your model's:

- Parameters
- Engine(s)
- Diffusion prescription
- Photosphere prescription
- SED prescription
- The photometry module

Optionally, your model file can also include an extinction prescription.

Then, you need to edit the ``parameters.json`` to include the priors on all ofyour model parameters. If no prior is specified, the variable will be set to a constant.

You can invoke the model using:

.. code-block:: bash

    python -m my_model_that_explains_everything


If your model requires a new engine, you can create this engine by again copying an existing engine:

.. code-block:: bash

	cp modules/engines/nickelcobalt.py my_new_engine.py

Then plug this engine into your model's json file.
