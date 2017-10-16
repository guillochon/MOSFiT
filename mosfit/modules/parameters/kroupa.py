"""Definitions for the `Kaussian` class."""
import numpy as np
from scipy.special import erfinv

from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class Kroupa(Parameter):
    """Parameter with Gaussian prior.

    If the parameter must be positive, set the `pos` keyword to True.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Kroupa, self).__init__(**kwargs)
        self._alpha1 = kwargs.get(self.key('alpha1'), None)
        self._alpha2 = kwargs.get(self.key('alpha2'), None)
        self._alpha3 = kwargs.get(self.key('alpha3'), None)
        self._x1 = kwargs.get(self.key('x1'), None)
        self._x2 = kwargs.get(self.key('x2'), None)

        if self._log:
            self._x1 = np.log(self._x1)
            self._x2 = np.log(self._x2)


        if not self._alpha1:
            raise ValueError('Need to set a value for alpha 1!')

        if not self._alpha2:
            raise ValueError('Need to set a value for alpha 2!')

        if not self._alpha3:
            raise ValueError('Need to set a value for alpha 3!')

        self._area_1 = (self._x1**(self._alpha1 + 1.0) - self._min_value**(self._alpha1 + 1.0))/(1.0 + self._alpha1)
        self._area_2 = self._x1**(self._alpha1 - self._alpha2) * ((self._x2**(self._alpha2 + 1.0) - self._x1**(self._alpha2 + 1.0))/(1.0 + self._alpha2))
        self._area_3 = self._x1**(self._alpha1 - self._alpha2) * self._x2**(self._alpha2 - self._alpha3) * (self._max_value**(self._alpha3 + 1.0) - self._x2**(self._alpha3 + 1.0))/(1.0 + self._alpha3)

        self._total_area = self._area_1 + self._area_2 + self._area_3

        self._C1 = 1.0/self._total_area
        self._C2 = 1.0 / self._total_area * self._x1**(self._alpha1 - self._alpha2)
        self._C3 = 1.0 / self._total_area * self._x1**(self._alpha1 - self._alpha2) * self._x2**(self._alpha2 - self._alpha3)


    def lnprior_pdf(self, x):
        """Evaluate natural log of probability density function."""
        value = self.value(x)
        if self._log:
            value = np.log(value)
        return self._C1 * self._alpha1 * np.log(value) * (value<self._x1) + \
                self._C2 * self._alpha2 * np.log(value) * (value>= self._x1) * (value<self._x2) + \
                self._C3 * self._alpha3 * np.log(value) * (value>= self._x2)

    def prior_cdf(self, u):
        """Evaluate inverse cumulative density function."""
        if u < self._area_1/self._total_area:
            value = (u * (self._alpha1 + 1.0)/self._C1 + self._min_value**(1. + self._alpha1)) ** (1./(1.+self._alpha1))
            #value = 10.0

        elif u < (self._area_2 + self._area_1)/self._total_area:
            u = u - self._area_1/self._total_area
            value = (u * (self._alpha2 + 1.0)/self._C2 + self._x1**(1. + self._alpha2)) ** (1./(1.+self._alpha2))
        else:
            u = u - (self._area_1 + self._area_2)/self._total_area
            value = (u * (self._alpha3 + 1.0)/self._C3 + self._x2**(1. + self._alpha3)) ** (1./(1.+self._alpha3))
        #raise ValueError(str(self._area_1),str(self._area_2/self._total_area+self._area_1/self._total_area),str(u),str(value))
        value = (value - self._min_value) / (self._max_value - self._min_value)
        #if not np.isfinite(value):
        #    value = 0.0

        return np.clip(value, 0.0, 1.0)
