"""Definitions for the `Kroupa` class."""
import numpy as np
from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class Kroupa(Parameter):
    """Kroupa IMF from https://arxiv.org/pdf/astro-ph/0102155.pdf.

    Defined for stellar masses larger than 0.01 solar masses.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Kroupa, self).__init__(**kwargs)

        if self._min_value > self._max_value:  # min mass must be < max mass
            raise ValueError(
                'Star mass prior must have max value > min value')
        if self._min_value < 0.01:
            raise ValueError(
                'Star mass kroupa prior is only defined down ' +
                'to 0.01 solar masses. Edit prior range or change prior.')

        self._norm = 1. / self.kroupa_cdf(self._min_value, self._max_value, 1)

    def lnprior_pdf(self, x):
        """Evaluate natural log of probability density function.

        PDF from Kroupa (2001b)
        """
        value = self.value(x)

        if value < 0.08:
            return np.log((value / 0.08)**(-0.3))
        elif value < 0.5:
            return np.log((value / 0.08)**(-1.3))
        else:
            return np.log((0.5 / 0.08)**(-1.3) * (
                value / 0.5)**(-2.3))

    def kroupa_cdf(self, minmass, maxmass, k):
        """Cumulative density function from Kroupa 2001b.

        0.01 <= mass, k is normalization
        """
        if minmass > maxmass:
            return 0  # useful default for calculating iCDF below
        elif minmass < 0.08:
            if maxmass < 0.08:
                prob = (k / 0.7 * 0.08**0.3 * (maxmass**0.7 - minmass**0.7))
            elif maxmass < 0.5:
                prob = ((k / 0.7 * 0.08**0.3 * (0.08**0.7 - minmass**0.7)) +
                        (k / -0.3) * 0.08**1.3 * (maxmass**(-0.3) -
                                                  0.08**(-0.3)))
            else:
                prob = ((k / 0.7 * 0.08**0.3 * (0.08**0.7 - minmass**0.7)) +
                        (k / -0.3) * 0.08**1.3 * (0.5**(-0.3) - 0.08**(-0.3)) +
                        (k / -1.3) * 0.08**1.3 * (0.5 * maxmass**(-1.3) -
                                                  0.5**(-0.3)))

        elif 0.08 <= minmass < 0.5:
            if maxmass < 0.5:
                prob = ((k / -0.3) * 0.08**1.3 * (maxmass**(-0.3) -
                        minmass**(-0.3)))
            else:  # maxmass > 0.5
                prob = (((k / -0.3) * 0.08**1.3 * (0.5**(-0.3) -
                        minmass**(-0.3))) +
                        ((k / -1.3) * 0.08**1.3 * 0.5 *
                        (maxmass**(-1.3) - 0.5**(-1.3))))

        elif 0.5 <= minmass:
            prob = ((k / -1.3) * 0.08**1.3 * 0.5 *
                    (maxmass**(-1.3) - minmass**(-1.3)))

        return prob

    def prior_icdf(self, u):
        """Evaluate inverse cumulative density function from Kroupa 2001b.

        output mass scaled to 0-1 interval
        """
        #  u < cdf(value = 0.08)
        if u < self.kroupa_cdf(self._min_value, 0.08, self._norm):  # if minvalue > maxvalue, CDF returns 0
            value = (u * 0.7 / self._norm * 0.08**(-0.3) +
                     self._min_value**0.7)**(1.0 / 0.7)
        #  cdf(value = 0.08) <= u < cdf(value = 0.5)
        elif u < self.kroupa_cdf(self._min_value, 0.5, self._norm):
            if self._min_value < 0.08:
                value = (((u - (self._norm / 0.7 * 0.08**0.3 *
                         (0.08**0.7 - self._min_value**0.7))) *
                        (-0.3) / self._norm * 0.08**(-1.3) +
                    0.08**(-0.3))**(1 / -0.3))
            else:  # 0.08 <= minvalue < 0.5
                value = (u * (-0.3 / self._norm) * 0.08**(-1.3) +
                         self._min_value**(-0.3))**(1/-0.3)
        #  cdf(value = 0.5) <= u
        else:
            if self._min_value < 0.08:
                value = ((u - (self._norm / 0.7 * 0.08**0.3 *
                         (0.08**0.7 - self._min_value**0.7)) -  # - CDF 0
                        ((self._norm / -0.3) * 0.08**1.3 *
                         (0.5**(-0.3) - 0.08**(-0.3)))) *  # - CDF 1
                         -1.3/self._norm * 0.08**(-1.3) * 2 +
                         0.5**(-1.3))**(1/-1.3)  # /relevant part of CDF 2

            elif self._min_value < 0.5:  # 0.08 <= self._min_value < 0.5
                value = ((u - ((self._norm / -0.3) * 0.08**1.3 *
                         (0.5**(-0.3) - self._min_value**(-0.3)))
                    ) * (-1.3/self._norm) * 0.08**(-1.3) * 2 +
                    0.5**(-1.3))**(1/-1.3)
            else:  # 0.5 <= self._min_value
                value = (u * (-1.3/self._norm) * 0.08**(-1.3) *
                         2 + self._min_value**(-1.3))**(1/-1.3)

        value = (value - self._min_value) / (self._max_value - self._min_value)
        # np.clip in case of python errors in line above
        return np.clip(value, 0.0, 1.0)
