from ..module import Module

CLASS_NAME = 'Photometry'


class Photometry(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._key = kwargs.get('key', '')
        self._data = kwargs.get('data', {})
        if self._data:
            name = list(self._data.keys())[0]
            photo = self._data[name]['photometry']
            self._times, self._magnitudes, self._e_magnitudes = list(
                map(list, zip(*[
                    [float(x['time']), float(x['magnitude']), float(x[
                        'e_magnitude'])] for x in photo
                    if 'time' in x and 'e_magnitude' in x
                ])))

    def process(self, **kwargs):
        if self._key == 'time':
            return {'times': self._times}
        elif self._key == 'magnitude':
            return {'magnitudes': self._magnitudes}
        elif self._key == 'e_magnitude':
            return {'e_magnitudes': self._e_magnitudes}
        return {}
