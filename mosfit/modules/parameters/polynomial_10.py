"""Definitions for the `polynomial` class."""
import numpy as np
from scipy.special import erfinv

from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class Polynomial10(Parameter):
    """Parameter with Gaussian prior.

    If the parameter must be positive, set the `pos` keyword to True.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Polynomial10, self).__init__(**kwargs)
        self._a = kwargs.get(self.key('a'), None)
        self._b = kwargs.get(self.key('b'), None)
        self._c = kwargs.get(self.key('c'), None)
        self._d = kwargs.get(self.key('d'), None)
        self._e = kwargs.get(self.key('e'), None)
        self._f = kwargs.get(self.key('f'), None)
        self._g = kwargs.get(self.key('g'), None)
        self._h = kwargs.get(self.key('h'), None)
        self._i = kwargs.get(self.key('i'), None)
        self._j = kwargs.get(self.key('j'), None)
        self._k = kwargs.get(self.key('k'), None)
        if self._log:
            self._a = np.log(self._a)

        if not self._a:
            raise ValueError('Need to set a value for mu!')

        if not self._b:
            raise ValueError('Need to set a value for sigma!')

    def lnprior_pdf(self, x):
        """Evaluate natural log of probability density function."""
        value = self.value(x)
        if self._log:
            value = np.log(value)

        values = np.linspace(self._min_value,self._max_value)
        p = np.poly1d([self._a,self._b,self._c,self._d,self._e,self._f,\
                self._g, self._h, self._i, self._j, self._k])
        normalization = 1./np.trapz(p(values),values)


        return np.log(normalization) + np.log(p(value))

    def prior_cdf(self, u):
        """Evaluate cumulative density function."""
        values = np.linspace(self._min_value,self._max_value)
        p = np.poly1d([self._a,self._b,self._c,self._d,self._e,self._f,\
                self._g, self._h, self._i, self._j, self._k])
        normalization = 1./np.trapz(p(values),values)

        cdf = np.cumtrapz(normalization*p(values),values)

        value = np.interp(u,cdf,values)

        print(value)
        sys.exit()


        value = (value - self._min_value) / (self._max_value - self._min_value)

        return np.clip(value, 0.0, 1.0)
