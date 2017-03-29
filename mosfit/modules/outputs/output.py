"""Definitions for the `Output` class."""
from mosfit.modules.module import Module

# Important: Only define one ``Module`` class per file.


class Output(Module):
    """Template class for output Modules."""

    def process(self, **kwargs):
        """Process module."""
        return {}
