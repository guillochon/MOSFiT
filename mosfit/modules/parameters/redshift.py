from astropy import units as un
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from mosfit.modules.parameters.parameter import Parameter

# Important: Only define one `Module` class per file.


class Redshift(Parameter):
    """Redshift parameter that depends on luminosity distance.
    """

    def process(self, **kwargs):
        """Initialize a parameter based upon either a fixed value or a
        distribution, if one is defined.
        """
        if (self._name in kwargs or self._min_value is None or
                self._max_value is None):
            # If this parameter is not free and is already set, then skip
            if self._name in kwargs:
                return {}

            self._lum_dist = kwargs.get('lumdist', None)
            if not self._value and self._lum_dist:
                value = z_at_value(cosmo.luminosity_distance,
                                   self._lum_dist * un.Mpc)
            elif self._value:
                value = self._value
            else:
                raise ValueError('Redshift has no value!')
        else:
            value = self.value(kwargs['fraction'])

        return {self._name: value}
