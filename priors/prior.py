"""Definitions for the `Prior` class."""
from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


class Prior(Module):
    """Generic prior module."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Prior, self).__init__(**kwargs)
