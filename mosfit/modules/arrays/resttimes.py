from mosfit.modules.module import Module

CLASS_NAME = 'RestTimes'


class RestTimes(Module):
    """This class converts the observed times to rest-frame times.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._times = kwargs['times'] if kwargs.get(
            'root', 'output') == 'output' else [
                x for x, y in zip(kwargs['times'], kwargs['observed']) if y
            ]
        self._bands = kwargs['bands'] if kwargs.get(
            'root', 'output') == 'output' else [
                x for x, y in zip(kwargs['bands'], kwargs['observed']) if y
            ]
        self._t_explosion = kwargs['texplosion']

        outputs = {}
        outputs['obsbands'] = self._bands
        outputs['resttimes'] = [
            x / (1.0 + kwargs['redshift']) for x in self._times
        ]
        outputs['resttexplosion'] = self._t_explosion / (
            1.0 + kwargs['redshift'])
        return outputs
