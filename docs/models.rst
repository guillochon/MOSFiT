.. _models:

Models in MOSFiT
================

``MOSFiT`` has been designed to be modular and easily modifiable by users
to alter, combine, and create physical models to approximate the behavior
of observed transients. On this page we walk through some basics on how a user
might alter an existing model shipped with the code, and how they could
go about creating their own model.

Altering an Existing Model
--------------------------

.. _altering_model

For many simple alterations to a model, such as adjusting input priors, setting
variables remain free and which should be fixed, and adding/removing modules
to the call stack, the user only needs to modify the JSON files that are copied to
their run directory. For these sorts of minor changes, no Python code should need to
be modified by the user!

For this example, we'll presume that the user will be modifying the ``slsn`` model.
First, the user should create a new run directory and run ``MOSFiT`` there once
to copy the model JSON files to that run directory::

    mkdir slsn_run
    cd slsn_run
    python -m mosfit -m slsn

After running, the user will notice three directories will have been copied to the run
directory: ``models``, ``jupyter``, and ``products``. Change into the ``models/slsn`` directory
and edit the ``parameters.json`` file in your favorite text editor::

    cd models/slsn
    vim parameters.json
    
You'll notice that ``parameters.json`` file is fairly bare-bones, containing only a list
of model parameters and their allowed value ranges.

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

    "Mns":{
        "min_value":1.5,
        "max_value":2.5
    }

**Congratulations!** You have just modified your first MOSFiT model. It should be
noted that even this very minor change, which affects the range of a single parameter,
would generate a completely different model hash than the default model, distinguishing
it from any other models that might have been uploaded by other users using the default settings.

Creating a New Model
--------------------

.. _creating_model

If users would like to create a brand new model for the ``MOSFiT`` platform, it is easiest to
duplicate one of the existing models that most closely resembles the model they
wish to create.
