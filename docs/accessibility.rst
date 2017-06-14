.. _accessibility:

=============
Accessibility
=============

.. _language:

--------
Language
--------

``MOSFiT`` can optionally translate all of its command line text into any language supported by Google translate. ``MOSFiT`` will use a user's ``$LANG`` environment variable to guess the language to use, if this variable is set. To accomplish this, the user must install the ``googletrans`` via ``pip``:

.. code-block:: bash

    pip install googletrans

Then, ``MOSFiT`` can be translated into one of the available languages using the ``--language`` option. When running for the first time for a new language, ``MOSFiT`` will pass all strings to the ``googletrans`` package one by one, which takes a few minutes to return the translated strings.

Note that Google's translation service is very approximate, and the translated text is only roughly equivalent to the original meaning.
