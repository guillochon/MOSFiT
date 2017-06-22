.. _faq:

==========================
Frequently Asked Questions
==========================

--------------------------------------------------
MOSFiT or one of its requirements isn't installing
--------------------------------------------------

We highly recommend using ``conda`` to install ``MOSFiT`` rather than ``pip``, as ``conda`` will skip some compilation steps that are common sources of error in the install process. If you are still having issues installing ``MOSFiT`` even with ``conda``, please ask us directly in the `MOSFiT Slack channel <https://astrochats.slack.com/messages/mosfit>`_.

-------------------------------------
The data MOSFiT is using is incorrect
-------------------------------------

If private data is not provided to ``MOSFiT``, it will draw data from the Open Astronomy Catalogs (the `Open Supernova Catalog <https://sne.space>`_ and the `Open Tidal Disruption Catalog <https://tde.space>`_). These catalogs are constructed by combining data from hundreds of individual sources, any one of which could have had an issue when being imported into the OACs. If you suspect the data contained for a transient on one of these catalogs is incorrect, please open an issue on the appropriate catalog repository (links to the repositories are available on the `AstroCats homepage <https://astrocats.space>`_) and the error will be corrected ASAP.

If you must correct the error immediately, feel free to copy the input file downloaded by ``MOSFiT`` (saved in a cache directory, the location of which is printed by ``MOSFiT`` when it runs) to your run directory and edit it on your own computer to fix the errors. But please *also* report the errors on the above issues pages so that the whole community will benefit!

-----------------------------------
Can I fit private data with MOSFiT?
-----------------------------------

Yes! Simply pass your ASCII datafile to the ``-e`` flag instead of the name of the transient you wish to fit. Your data will remain private unless you choose to upload it with the optional ``-u`` flag, which will warn you before any data is uploaded publicly. More info on fitting private data can be found :ref:`here <private>`.

------------------------------------------------------------------
How do I exclude particular instruments/bands/sources from my fit?
------------------------------------------------------------------

Excluding instruments can be accomplished by using the ``--exclude-instruments`` option, and excluding bands can be accomplished using the ``--exclude-bands`` option. All the data from a particular source (e.g. a paper or survey) can be excluded using ``--exclude-sources`` (see :ref:`here <restricting>` for more information on restricting your dataset). More complicated exclusion rules (say ignoring a particular band from a particular instrument, but not for other instruments) are most easily accomplished by simply deleting the unwanted data from the input file; users should copy the cached version downloaded from the Open Astronomy Catalogs to their run directory and edit the files to remove the data.
