"""Definitions for the `LightCurve` class."""
from mosfit.modules.outputs.output import Output


# Important: Only define one ``Module`` class per file.


class Write(Output):
    """Write keys to disk."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Write, self).__init__(**kwargs)

    def process(self, **kwargs):
        """Process module."""
        # Dummy function for now, not implemented.
        return {}
