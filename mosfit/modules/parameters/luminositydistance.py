"""Definitions for the `LuminosityDistance` class."""
import numpy as np
from astropy import units as un
from astropy.cosmology import Planck15 as cosmo

from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class LuminosityDistance(Parameter):
    """LuminosityDistance parameter that depends on luminosity distance."""

    def process(self, **kwargs):
        """Process module."""
        # If this parameter is not free and is already set, then skip
        if self._name in kwargs:
            return {}

        if self._value is None:
            self._redshift = kwargs.get(self.key('redshift'), None)
            if self._redshift is not None:
                if self._redshift < 0.0:
                    msg = self._printer.message(
                        'negative_redshift', [
                            str(np.around(self._redshift, decimals=2))],
                        prt=False)
                    raise(RuntimeError(msg))
                else:
                    value = (cosmo.luminosity_distance(
                        self._redshift) / un.Mpc).value
            else:
                value = self.value(kwargs['fraction'])
        else:
            value = self._value

        return {self._name: value}
