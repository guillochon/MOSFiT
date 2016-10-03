import numpy as np
from mosfit.modules.module import Module

CLASS_NAME = 'DenseTimes'


class DenseTimes(Module):
    """This class ensures an even time-sampling between the time of explosion
    and the last datapoint, as many transients may lack regular candence data.
    """

    N_TIMES = 100
    L_T_MIN = 1.0e-3  # in days

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_times = kwargs[
            'n_times'] if 'n_times' in kwargs else self.N_TIMES

    def process(self, **kwargs):
        self._times = kwargs['times']
        self._t_explosion = kwargs['texplosion']

        outputs = {}
        max_times = max(kwargs['times'])
        if max_times > kwargs['texplosion']:
            outputs['densetimes'] = list(
                sorted(
                    set([0.0] + [
                        x + self._t_explosion
                        for x in np.logspace(
                            self.L_T_MIN,
                            max_times - self._t_explosion,
                            num=self._n_times)
                    ] + self._times)))
        else:
            outputs['densetimes'] = kwargs['times']
        return outputs
