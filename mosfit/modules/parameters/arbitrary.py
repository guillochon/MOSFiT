"""Definitions for the `arbitrary` class."""
from scipy import interpolate
from scipy.integrate import quad
from astropy.cosmology import WMAP9 as cosmo
import numpy as np
import pandas as pd

from mosfit.modules.parameters.parameter import Parameter

# Important: Only define one ``Module`` class per file.

class Arbitrary(Parameter):
    """Parameter with Arbitrary prior.

        Contribution by Peter S. H. Cheung"""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Arbitrary, self).__init__(**kwargs)
        self._filename = kwargs.get(self.key('filename'), None)

        """Input data from file""" #Put the axis label (X,Y) on the first row of file
        df = pd.read_csv(self._filename, header = 0)

        """Data treatment""" #Ignore this part if x and y are ready to fit.
        df.Y = 4 * np.pi * 10 ** df.Y * cosmo.differential_comoving_volume(df.X) 
        
        """Generating distribution shape and normalizing factor"""
        self._min_value = self._min_value if self._min_value else df.X.min()
        self._max_value = self._max_value if self._max_value else df.X.max()
        self._f = interpolate.interp1d(df.X, df.Y, kind = 'linear', fill_value="extrapolate")
        norm_factor = quad(self._f, self._min_value, self._max_value)[0]
        
        """Generating icdf"""
        self._cdf = []
        self._x = np.linspace(self._min_value, self._max_value, 100)
        for i in self._x:
            integral = quad(self._f, 0.1, i)[0]
            self._cdf.append(integral / norm_factor)
        self._icdf = interpolate.interp1d(self._cdf, self._x, kind = 'linear', fill_value = 'extrapolate')

    def lnprior_pdf(self, x):
        """Evaluate natural log of probability density function."""
        value = self.value(x)
        if self._log:
            value = np.log(value)
        return self._f(value)

    def prior_icdf(self, u):
        """Evaluate inverse cumulative density function."""
        value = self._icdf(u)
        value = (value - self._min_value) / (self._max_value - self._min_value)
        return value
