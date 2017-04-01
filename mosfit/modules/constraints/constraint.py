"""Definitions for the `Constraint` class."""
from collections import OrderedDict

from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


class Constraint(Module):
    """Template class for constraints."""

    def process(self, **kwargs):
        """Process module."""
        self._score_modifier = 0.0
        return OrderedDict([(self.key('score_modifier'),
                             self._score_modifier)])
