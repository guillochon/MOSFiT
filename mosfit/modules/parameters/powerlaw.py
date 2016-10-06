import numpy as np
from mosfit.modules.parameters.parameter import Parameter

CLASS_NAME = 'PowerLaw'


class PowerLaw(Parameter):

    "Standard power law, alpha must be > 1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._alpha = kwargs.get('alpha', None)
        # Scale to min/max_value
        if self._log:
            miv = np.exp(self._min_value)
            mav = np.exp(self._max_value)
        else:
            miv = self._min_value
            mav = self._max_value
        self._mivap1 = miv**(self._alpha + 1.0)
        self._mavap1 = mav**(self._alpha + 1.0)
        self._miavap1 = self._mavap1 - self._mivap1
        self._cdf_exp = 1.0 / (self._alpha + 1.0)

    def lnprior_pdf(self, value):
        return np.log(self._alpha / self._min_value) - self._alpha * np.log(
            value / self._min_value)

    def prior_cdf(self, y):
        value = ((self._mivap1 + y * self._miavap1)**self._cdf_exp)
        if self._log:
            value = np.log(value)
        value = (value - self._min_value) / (self._max_value - self._min_value)
        return value
