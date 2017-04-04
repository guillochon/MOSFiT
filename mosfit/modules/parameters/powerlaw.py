"""Definitions for the `PowerLaw` class."""
import numpy as np

from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class PowerLaw(Parameter):
    """Standard power law, alpha must be > 1."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(PowerLaw, self).__init__(**kwargs)
        self._alpha = kwargs.get(self.key('alpha'), None)
        if self._log:
            self._miv = np.exp(self._min_value)
            self._mav = np.exp(self._max_value)
        else:
            self._miv = self._min_value
            self._mav = self._max_value
        self._mivap1 = self._miv ** (self._alpha + 1.0)
        self._mavap1 = self._mav ** (self._alpha + 1.0)
        self._miavap1 = self._mavap1 - self._mivap1
        self._cdf_exp = 1.0 / (self._alpha + 1.0)

    def lnprior_pdf(self, x):
        """Evaluate natural log of probability density function."""
        value = self.value(x)
        return np.log(((value - self._miv) / (self._mav - self._miv)) **
                      self._alpha)

    def prior_cdf(self, u):
        """Evaluate cumulative density function."""
        value = ((self._mivap1 + u * self._miavap1) ** self._cdf_exp)
        if self._log:
            value = np.log(value)
        value = (value - self._min_value) / (self._max_value - self._min_value)
        return value
