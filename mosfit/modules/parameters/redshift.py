"""Definitions for the `Redshift` class."""
import numpy as np
from astropy import units as un
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value

from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class Redshift(Parameter):
    """Redshift parameter that depends on luminosity distance."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Redshift, self).__init__(**kwargs)

        self._warned_small = False

    def process(self, **kwargs):
        """Process module."""
        # If this parameter is not free and is already set, then skip
        if self._name in kwargs:
            return {}

        if self._value is None:
            self._lum_dist = kwargs.get(self.key('lumdist'), self._lum_dist)
            if self._value is None and self._lum_dist is not None:
                if self._lum_dist < 1.0:
                    if not self._warned_small:
                        self._printer.message(
                            'small_lumdist', [
                                str(np.around(
                                    self._lum_dist * 1.0e6, decimals=2))],
                            warning=True)
                    self._warned_small = True
                    value = 0.0
                else:
                    value = z_at_value(cosmo.luminosity_distance,
                                       self._lum_dist * un.Mpc)
            else:
                value = self.value(kwargs['fraction'])
        else:
            value = self._value

        return {self._name: value}

    def send_request(self, request):
        """Send requests to other modules."""
        if request == 'redshift':
            return self._value

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        self._lum_dist = requests.get('lumdist', None)
