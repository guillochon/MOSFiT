"""Definitions for the `Constant` class."""
from mosfit.modules.parameters.parameter import Parameter

# Important: Only define one ``Module`` class per file.


class Constant(Parameter):
    """Constant parameter.

    `Parameter` that will throw an error if the user attempts to make the
    variable free.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Constant, self).__init__(**kwargs)
        if self._min_value is not None or self._max_value is not None:
            raise ValueError('`Constant` class cannot be assigned minimum and '
                             'maximum values!')
