.. _help:

====
Help
====

.. _faq:

--------------------------
Frequently Asked Questions
--------------------------

What do I do if MOSFiT or one of its requirements isn't installing?
===================================================================

We highly recommend using ``conda`` to install ``MOSFiT`` rather than ``pip``, as ``conda`` will skip some compilation steps that are common sources of error in the install process. If you are still having issues installing ``MOSFiT`` even with ``conda``, please ask us directly in the `#mosfit Slack channel on AstroChats <https://slack.astrocats.space>`_.

What can I try if MOSFiT won't run?
===================================

If ``MOSFiT`` is the first ``conda`` program you've used, and you previously used your system's built-in Python install, your shell environment may still be set up for your old Python setup, which can cause problems both for ``MOSFiT`` and your old Python programs. One common issue is that your ``PYTHON_PATH`` environment variable might be set to your build-in Python's install location, this will supercede conda's paths and potentially cause issues. Edit your ``.bashrc`` or ``.profile`` file to remove any ``PYTHON_PATH`` variable declarations, this will prevent path conflicts.

Is MOSFiT using the correct data?
=================================

If private data is not provided to ``MOSFiT``, it will draw data from the Open Astronomy Catalogs (the `Open Supernova Catalog <https://sne.space>`_ and the `Open Tidal Disruption Catalog <https://tde.space>`_). These catalogs are constructed by combining data from hundreds of individual sources, any one of which could have had an issue when being imported into the OACs. If you suspect the data contained for a transient on one of these catalogs is incorrect, please open an issue on the appropriate catalog repository (links to the repositories are available on the `AstroCats homepage <https://astrocats.space>`_) and the error will be corrected ASAP.

If you must correct the error immediately, feel free to copy the input file downloaded by ``MOSFiT`` (saved in a cache directory, the location of which is printed by ``MOSFiT`` when it runs) to your run directory and edit it on your own computer to fix the errors. But please *also* report the errors on the above issues pages so that the whole community will benefit!

Can I fit private data with MOSFiT?
===================================

Yes! Simply pass your ASCII datafile to the ``-e`` flag instead of the name of the transient you wish to fit. Your data will remain private unless you choose to upload it with the optional ``-u`` flag, which will warn you before any data is uploaded publicly. More info on fitting private data can be found :ref:`here <private>`.

How do I exclude particular instruments/bands/sources from my fit?
==================================================================

Excluding instruments can be accomplished by using the ``--exclude-instruments`` option, and excluding bands can be accomplished using the ``--exclude-bands`` option. All the data from a particular source (e.g. a paper or survey) can be excluded using ``--exclude-sources`` (see :ref:`here <restricting>` for more information on restricting your dataset). More complicated exclusion rules (say ignoring a particular band from a particular instrument, but not for other instruments) are most easily accomplished by simply deleting the unwanted data from the input file; users should copy the cached version downloaded from the Open Astronomy Catalogs to their run directory and edit the files to remove the data.

.. _contact:

-------
Contact
-------

If you need additional help, the most rapid way to receive it is to join `our Slack channel <https://astrochats.slack.com/messages/mosfit>`_. Barring that, feel free to `contact us via e-mail <mailto:guillochon@gmail.com>`_.
