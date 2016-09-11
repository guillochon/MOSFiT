import numpy as np

from ..module import Module

CLASS_NAME = 'Parameter'


class Parameter(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_value = kwargs.get('max_value', None)
        self._min_value = kwargs.get('min_value', None)
        self._value = kwargs.get('value', None)
        self._log = kwargs.get('log', False)
        if (self._log and self._min_value is not None and
                self._max_value is not None):
            self._min_value = np.log(self._min_value)
            self._max_value = np.log(self._max_value)

    def process(self, **kwargs):
        if self._min_value is None or self._max_value is None:
            value = self._value
        else:
            value = max(
                min((kwargs['fraction'] *
                     (self._max_value - self._min_value) + self._min_value),
                    self._max_value), self._min_value)
            if self._log:
                value = np.exp(value)
        return {self._name: value}
