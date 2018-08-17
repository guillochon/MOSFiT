.. _installation:

============
Installation
============

Several installation methods for ``MOSFiT`` are outlined below. If you run into
issues, `open a new issue <https://github.com/guillochon/mosfit/issues>`_ on
GitHub.

.. _anaconda:

-------------------------------
Setting up MOSFiT with Anaconda
-------------------------------

**Platforms:** MacOS X, Linux, and Windows

We recommend using the `Anaconda <http://continuum.io/downloads.html>`__ Python
distribution from Continuum Analytics (or the related Miniconda distribution)
as your Python environment.

After installing conda, ``MOSFiT`` can be installed via:

.. code-block:: bash

    conda install -c conda-forge mosfit

.. _pip:

-------------------
Installing with pip
-------------------

**Platforms:** MacOS X, Linux, and Windows

Installing ``MOSFiT`` with pip is straightforward:

.. code-block:: bash

    pip install mosfit

.. _source:

----------------------
Installing from source
----------------------

**Platforms:** MacOS X, Linux, and Windows

If you are interested in performing more serious development work, it is probably best to install ``MOSFiT`` from source. This can be done by cloning the repository and then running the ``setup.py`` script:

.. code-block:: bash

    git clone https://github.com/guillochon/MOSFiT.git
    cd MOSFiT
    python setup.py install
