import numpy as np
from mosfit.modules.parameters.parameter import Parameter

CLASS_NAME = 'PowerLaw'


class PowerLaw(Parameter):

    "Standard power law, alpha must be > 1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._alpha = kwargs.get('alpha', None)
        self._x_max = kwargs.get('xmax')

    def lnprior_pdf(self, value):
        return np.log(self._alpha / self._x_min) - self._alpha * np.log(
            value / self._x_min)

    def prior_cdf(self, **kwargs):
        return (1.0 - kwargs['fraction'])**(1. /
                                            (1.0 - self._alpha)) * self._x_max
