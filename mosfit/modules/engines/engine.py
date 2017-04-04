"""Definitions for the `Engine` class."""
from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


class Engine(Module):
    """Generic engine module."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Engine, self).__init__(**kwargs)
        self._wants_dense = True
