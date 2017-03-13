"""Definitions for the `Covariance` class."""

from mosfit.modules.parameters.parameter import Parameter

# Important: Only define one ``Module`` class per file.


class Covariance(Parameter):
    """Model parameter that can either be free or fixed."""
