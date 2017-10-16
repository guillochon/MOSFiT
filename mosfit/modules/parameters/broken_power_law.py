"""Definitions for the `BrokenPowerLaw` class."""
import numpy as np

from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class BrokenPowerLaw(Parameter):
    """Broken power law"""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(BrokenPowerLaw, self).__init__(**kwargs)
        self._alpha1 = kwargs.get(self.key('alpha1'), None)
        self._alpha2 = kwargs.get(self.key('alpha2'), None)
        self._vbreak = kwargs.get(self.key('vbreak'), None)
        self._miv = self._min_value
        self._mav = self._max_value


        self._A = (((1. + self._vbreak)**(self._alpha1 + 1.0) - (1. + self._miv)**(self._alpha1 + 1.0)) / (1. + self._alpha1) + \
                  (1. + self._vbreak)**(self._alpha1 - self._alpha2) * ((1. + self._mav)**(1. + self._alpha2) - (1. + self._vbreak)**(1. + self._alpha2)) / (1. + self._alpha2))**-1.0
        self._B = self._A * (1. + self._vbreak)**(self._alpha1 - self._alpha2)

    def lnprior_pdf(self, x):
        """Evaluate natural log of probability density function."""
        value = self.value(x)
        return self._alpha1 * np.log(self._A * (1. + x)) * (x<=self._vbreak) + \
               np.log((self._A * (1. + self._vbreak)**(self._alpha1) + self._B * (1. + x)**(self._alpha2))) * (x>self._vbreak)

    def prior_cdf(self, u):
        """Evaluate inverse cumulative density function."""

        value = (u - self._min_value) / (self._max_value - self._min_value)
        value = u
        cdf_break = self._A / (1.0 + self._alpha1) * ((1.0 + self._vbreak)**(1.0 + self._alpha1) - (1.0 + self._miv)**(1.0 + self._alpha1))

        inv_cdf = ((value * (1. + self._alpha1)/self._A + (1. + self._miv)**(1. + self._alpha1))**(1./(1. + self._alpha1)) - 1.0) * (value <= cdf_break) + \
                  (((value -cdf_break) * (1.0 + self._alpha2)/self._B + (1.0 + self._vbreak)**(1.0 + self._alpha2))**(1.0/(1.0 + self._alpha2)) - 1.0) * (value > cdf_break)


        inv_cdf = (inv_cdf - self._min_value) / (self._max_value - self._min_value)

        return inv_cdf
