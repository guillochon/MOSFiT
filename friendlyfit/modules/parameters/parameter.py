from ..module import Module

CLASS_NAME = 'Parameter'


class Parameter(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_value = kwargs.get('max_value', None)
        self._min_value = kwargs.get('min_value', None)
        self._value = kwargs.get('value', None)
        self._log = kwargs.get('log', False)

    def process(self, **kwargs):
        if not self._min_value or not self._max_value:
            value = self._value
        else:
            value = max(
                min((kwargs['fraction'] *
                     (self._max_value - self._min_value) + self._min_value),
                    self._max_value), self._min_value)
        return {self._name: value}
