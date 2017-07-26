"""Definitions for the `LuminosityDistance` class."""
import numpy as np
from astropy import units as un
from astropy.cosmology import Planck15 as cosmo

from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class LuminosityDistance(Parameter):
    """LuminosityDistance parameter that depends on luminosity distance."""

    def __init__(self, **kwargs):
        """Initialize module."""
        super(LuminosityDistance, self).__init__(**kwargs)

        self._warned_small = False

    def process(self, **kwargs):
        """Process module."""
        # If this parameter is not free and is already set, then skip
        if self._name in kwargs:
            return {}

        if self._value is None:
            self._redshift = kwargs.get(self.key('redshift'), self._redshift)
            if self._redshift is not None:
                if self._redshift <= 0.0:
                    if not self._warned_small:
                        self._printer.message(
                            'negative_redshift', [
                                str(np.around(self._redshift, decimals=2))],
                            warning=True)
                    self._warned_small = True
                    value = 1.0e-5
                else:
                    value = (cosmo.luminosity_distance(
                        self._redshift) / un.Mpc).value
            else:
                value = self.value(kwargs['fraction'])
        else:
            value = self._value

        return {self._name: value}

    def send_request(self, request):
        """Send requests to other modules."""
        if request == 'lumdist':
            return self._value

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        self._redshift = requests.get('redshift', None)
