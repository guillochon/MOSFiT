"""Definitions for the `Utility` class."""
from collections import OrderedDict

from mosfit.modules.utilities.utility import Utility


# Important: Only define one ``Module`` class per file.


class Rename(Utility):
    """Template class for photosphere Modules."""

    def process(self, **kwargs):
        """Process module."""
        output = OrderedDict([('_delete_keys', [])])
        for rep in self._replacements:
            for key in kwargs:
                if rep in key:
                    output['_delete_keys'].append(key)
                    data = kwargs[key]
                    output[key.replace(rep, self._replacements[rep])] = data
        return output
