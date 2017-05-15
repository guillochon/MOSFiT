.. _models:

Models in MOSFiT
================

``MOSFiT`` has been designed to be modular and easily modifiable by users
to alter, combine, and create physical models to approximate the behavior
of observed transients. On this page we cover some basics on how a user
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

Creating a New Model
--------------------

.. _creating_model

If users would like to create a brand new model for the ``MOSFiT`` platform, it is easiest to
duplicate one of the existing models that most closely resembles the model they
wish to create.
