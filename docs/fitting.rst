.. _fitting:

======================
Fitting models to data
======================

The primary purpose of ``MOSFiT`` is to fit models of transients to observed data. In this section we cover "how" of fitting, and how the user should interpret their results.

-----------
Public data
-----------

.. _public

``MOSFiT`` is deeply connected to the Open Catalogs (The Open Supernova Catalog, the Open Tidal Disruption Catalog, etc.), and the user can directly fit their model against any data provided by those catalogs. The Open Catalogs store names for each transient, and the user can access any transient by any known name of that transient. As an example, both of the commands below will fit the same transient:: bash

    mosfit -m slsn -e PTF11dij
    mosfit -m slsn -e CSS110406:135058+261642

While the Open Catalogs do their best to maintain the integrity of the data they contain, there is always the possibility that the data contains errors, so users are encouraged to spot check the data they download before using it for any scientific purpose. A common error is that the data has been tagged with the wrong photometric system, or has not been tagged with a photometric system at all and uses a different system from what is commonly used for a given telescope/instrument/band. Users are encouraged to immediately report any issues with the public data on the GitHub issues page assocated with that catalog (e.g. the Open Supernova Catalog's `issue page <https://github.com/astrocatalogs/supernovae/issues>`_).

------------
Private data
------------

.. _private

If you have private data you would like to fit, the most robust way to load the data into ``MOSFiT`` is to directly construct a JSON file from your data that conforms to the `Open Catalog Schema <https://github.com/astrocatalogs/supernovae/blob/master/SCHEMA.md>`_. This way, the user can specify all the data that ``MOSFiT`` can use for every single observation in a precise way. All data provided by the Open Catalogs is provided in this form, and if the user open up a typical JSON file downloaded from one of these catalogs, they will find that each observation is tagged with all the information necessary to model it.

Of course, it is more likely, that the data a user will have handy will be in another form, typically an ASCII table where each row presents a single (or multiple) observations. ``MOSFiT`` includes a conversion feature where the user can simply pass the path to the file(s) to convert:

.. code-block:: bash

    mosfit -e path/to/my/ascii/file/my_transient.dat
    mosfit -e path/to/my/folder/of/ascii/files/*.dat

In some cases, if the ASCII file is in a simple form with columns that match all the required columns, ``MOSFiT`` will silently convert the input files into JSON files, a copy of which will be saved to the current run directory. In most cases however, the user will be prompted to answer a series of questions about the data in a "choose your own adventure" style. If passed a list of files, ``MOSFiT`` will assume all the files share the same format and the user will only be asked questions about the first file.

If the user so chooses, they may *optionally* upload their data directly to the Open Catalogs with the ``-u`` flag. This will make their observational data publicly accessible on the Open Catalogs::

    mosfit -e path/to/my/ascii/file/my_transient.dat -u

Note that this step is completely optional, users do not have to share their data publicly to use ``MOSFiT``. If a user believes they have uploaded any private data in error, they are encouraged to immediately contact the maintainers_.
