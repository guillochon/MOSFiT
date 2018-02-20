"""Definitions for the `Library` class."""
from math import isnan

import numpy as np

from mosfit.modules.arrays.array import Array
from mosfit.utils import flux_density_unit


# Important: Only define one ``Module`` class per file.


class Library(Array):
    """Calculate the diagonal/residuals for a model kernel."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Library, self).__init__(**kwargs)

    def process(self, **kwargs):
        """Process module."""
        return {'library': self._library}
