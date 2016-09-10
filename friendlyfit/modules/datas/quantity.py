from ..module import Module

CLASS_NAME = 'Quantity'


class Quantity(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._key = kwargs.get('key', '')

    def process(self, **kwargs):
        return {self._key: self._value}

    def set_data(self, data):
        if data:
            name = list(data.keys())[0]
            self._value = float(data[name][self._key][0]['value'])
