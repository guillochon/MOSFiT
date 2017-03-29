"""Definitions for the `Output` class."""
from mosfit.modules.module import Module

# Important: Only define one ``Module`` class per file.


class Output(Module):
    """Template class for output Modules."""

    def set_attributes(self, task):
        """Set module attributes based on task specification."""
        super(Output, self).set_attributes(task)
        self._output_keys = task.get('output_keys', [])

    def process(self, **kwargs):
        """Process module."""
        return {}
