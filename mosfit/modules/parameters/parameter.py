import numpy as np

from ..module import Module

CLASS_NAME = 'Parameter'


class Parameter(Module):
    """Model parameter that can either be free or fixed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_value = kwargs.get('max_value', None)
        self._min_value = kwargs.get('min_value', None)
        self._value = kwargs.get('value', None)
        self._log = kwargs.get('log', False)
        self._latex = kwargs.get('latex', self._name)
        if (self._log and self._min_value is not None and
                self._max_value is not None):
            self._min_value = np.log(self._min_value)
            self._max_value = np.log(self._max_value)

    def latex(self):
        return self._latex

    def process(self, **kwargs):
        """Initialize a parameter based upon either a fixed value or a
        distribution, if one is defined.
        """

        if self._min_value is None or self._max_value is None:
            # If this parameter is not free and is already set, then skip
            if self._name in kwargs:
                return {}

            value = self._value
        else:
            value = max(
                min((kwargs['fraction'] *
                     (self._max_value - self._min_value) + self._min_value),
                    self._max_value), self._min_value)
            if self._log:
                value = np.exp(value)
        return {self._name: value}
