from math import erf
import numpy as np
from mosfit.modules.parameters.parameter import Parameter
from scipy.special import erfinv

CLASS_NAME = 'Gaussian'


class Gaussian(Parameter):

    """
    Gaussian Prior

    If the parameter much be positive, set the pos keyword to 1

    """

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._mu = kwargs.get('mu', None)
        self._sigma = kwargs.get('sigma',None)
        self._pos = kwargs.get('pos',0)

        if not self._mu:
            raise ValueError('Need to set a value for mu')

        if not self._sigma:
            raise ValueError('Need to set a value for sigma')

    def lnprior_pdf(self,value):
        return np.log(1./(np.sqrt(4. * np.pi) * self._sigma)) - (value - self._mu)**2 \
                                                         / (2. * self._sigma**2)

    def prior_cdf(self,**kwargs):
        value = (erfinv(2.0 * kwargs['fraction'] - 1.0) * np.sqrt(2.)) * self._sigma + self._mu

        if self._pos == 1:
            return max(value,0.0)

        else:
            return value

