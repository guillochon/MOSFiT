import numpy as np
from mosfit.modules.parameters.parameter import Parameter
from scipy.special import erfinv

# Important: Only define one `Module` class per file.


class Gaussian(Parameter):
    """
    Gaussian Prior

    If the parameter must be positive, set the pos keyword to True

    """

    def __init__(self, **kwargs):
        super(Gaussian, self).__init__(**kwargs)
        self._mu = kwargs.get('mu', None)
        self._sigma = kwargs.get('sigma', None)
        if self._log:
            self._mu = np.log(self._mu)
            self._sigma = np.log(10.0**self._sigma)

        if not self._mu:
            raise ValueError('Need to set a value for mu!')

        if not self._sigma:
            raise ValueError('Need to set a value for sigma!')

    def lnprior_pdf(self, x):
        value = self.value(x)
        if self._log:
            value = np.log(value)
        return -(value - self._mu)**2 / (2. * self._sigma**2)

    def prior_cdf(self, u):
        value = (erfinv(2.0 * u - 1.0) * np.sqrt(2.)) * self._sigma + self._mu
        value = (value - self._min_value) / (self._max_value - self._min_value)

        return np.clip(value, 0.0, 1.0)
