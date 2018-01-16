"""Definitions for the `Kroupa` class."""
import numpy as np
from mosfit.modules.parameters.parameter import Parameter


# Important: Only define one ``Module`` class per file.


class Kroupa(Parameter):
    """Kroupa IMF from https://arxiv.org/pdf/astro-ph/0102155.pdf.

    Requires that min_value = 0.01 solar masses.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Kroupa, self).__init__(**kwargs)
        self._norm = 1. / self.kroupa_cdf(self._max_value, 1)

    def lnprior_pdf(self, x):
        """Evaluate natural log of probability density function.

        PDF from Kroupa (2001b)
        """
        value = self.value(x)

        if value < 0.08:
            return np.log(self._norm * (x / 0.08)**(-0.3))
        elif value < 0.5:
            return np.log(self._norm * (x / 0.08)**(-1.3))
        else:
            return np.log(self._norm * (0.5 / 0.08)**(-1.3) * (
                x / 0.5)**(-2.3))

    def kroupa_cdf(self, maxmass, k):
        """Cumulative density function from Kroupa 2001b.

        0.01 <= mass, k is normalization
        """
        if maxmass < 0.01:
            return None
        elif maxmass < 0.08:
            return (k / 0.7 * 0.08**0.3 * (maxmass**0.7 - 0.01**0.7))
        elif maxmass < 0.5:
            return ((k / 0.7 * 0.08**0.3 * (0.08**0.7 - 0.01**0.7)) +
                    (k / -0.3) * 0.08**1.3 * (maxmass**(-0.3) -
                                              0.08**(-0.3)))
        else:
            return ((k / 0.7 * 0.08**0.3 * (0.08**0.7 - 0.01**0.7)) +
                    (k / -0.3) * 0.08**1.3 * (0.5**(-0.3) - 0.08**(-0.3)) +
                    (k / -1.3) * 0.08**1.3 * (0.5 * maxmass**(-1.3) -
                                              0.5**(-0.3)))

    def prior_cdf(self, u):
        """Inverse cumulative density function from Kroupa 2001b.

        output mass scaled to 0-1 interval
        min mass before scaling = 0.01
        """
        if u < self.kroupa_cdf(0.08, self._norm):
            value = (u * (0.7) / self._norm * 0.08**(-0.3) +
                     0.01**0.7)**(1 / 0.7)
        elif u < self.kroupa_cdf(0.5, self._norm):
            value = (((u - (self._norm / 0.7 * 0.08**0.3 *
                            (0.08**0.7 - 0.01**0.7))) * (-0.3) / self._norm *
                      0.08**(-1.3) + 0.08**(-0.3))**(1 / -0.3))
        else:
            value = (((u - (self._norm / -0.3) * 0.08**1.3 *
                       (0.5**(-0.3) - 0.08**(-0.3)) -
                       (self._norm / 0.7 * 0.08**0.3 *
                        (0.08**0.7 - 0.01**0.7))) * -1.3 / self._norm *
                      0.5**(-2.3) * (6.25)**1.3 + 0.5**(-1.3))**(1 / -1.3))

        value = (value - self._min_value) / (self._max_value - self._min_value)
        # np.clip in case of python errors in line above
        return np.clip(value, 0.0, 1.0)
