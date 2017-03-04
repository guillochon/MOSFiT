from mosfit.modules.module import Module

# Important: Only define one `Module` class per file.


class RestTimes(Module):
    """This class converts the observed times to rest-frame times.
    """

    def process(self, **kwargs):
        self._times = kwargs['all_times']
        self._t_explosion = kwargs['texplosion']

        outputs = {}
        outputs['rest_times'] = [
            x / (1.0 + kwargs['redshift']) for x in self._times
        ]
        outputs['resttexplosion'] = self._t_explosion / (
            1.0 + kwargs['redshift'])
        return outputs
