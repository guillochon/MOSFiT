"""Definitions for the `Engine` class."""
from mosfit.modules.module import Module


# Important: Only define one ``Module`` class per file.


class Engine(Module):
    """Generic engine module."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Engine, self).__init__(**kwargs)
        self._wants_dense = True

    def add_to_existing_lums(self, new_lums, **kwargs):
        """Add luminosities from module to existing ``luminosities`` key.

        Parameters
        ----------
        new_lums : array
            The new luminosities to add to the existing luminosities.

        Returns
        -------
        new_lums : array
            The result of summing the new and existing luminosities.
        """
        # Add on to any existing luminosity
        old_lums = kwargs.get('dense_luminosities', None)
        if old_lums is not None:
            new_lums = [x + y for x, y in zip(old_lums, new_lums)]
        return new_lums
