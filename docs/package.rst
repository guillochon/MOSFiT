.. _package:

==================
Using as a package
==================

If you wish to produce light curves or other data products for a given model without using the fitting and evidence accumulation features of ``MOSFiT``, functions within the code can be accessed by importing the ``mosfit`` package into your Python code.

.. _run:

---------------------
Produce model outputs
---------------------

In the code snippet below, we fetch a supernova's data from the Open Catalogs using the ``Fetcher`` class, create a ``Model`` that initializes from the fetched data, and finally run the model:

.. code-block:: python

    import mosfit
    import numpy as np

    # Create an instance of the `Fetcher` class.
    my_fetcher = mosfit.fetcher.Fetcher()

    # Fetch some data from the Open Supernova Catalog.
    fetched = my_fetcher.fetch('SN2009do')[0]

    # Instantiatiate the `Model` class (selecting 'slsn' as the model).
    my_model = mosfit.model.Model(model='slsn')

    # Load the fetched data into the model.
    my_model.load_data(my_fetcher.load_data(fetched), event_name=fetched['name'])

    # Generate a random input vector of free parameters.
    x = np.random.rand(my_model.get_num_free_parameters())

    # Produce model output.
    outputs = my_model.run(x)
    print('Keys in output: `{}`'.format(', '.join(list(outputs.keys()))))
