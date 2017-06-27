"""Definitions for the `Kroupa` class."""
from math import pi
# import numexpr as ne
import numpy as np
from astropy import constants as c
from mosfit.constants import DAY_CGS, KM_CGS, M_SUN_CGS, C_CGS  # FOUR_PI
from priors.prior import Prior
# from scipy.interpolate import interp1d


class Kroupa(Prior):
    """Kroupa IMF prior.

    Currently require starmass range to be 0.01 to 100
    https://arxiv.org/pdf/astro-ph/0102155.pdf
    """

    def __init__(self, **kwargs):
        """Initialize module.

        Calculates normalization factor
        for mass generating function.
        """
        super(Kroupa, self).__init__(**kwargs)

        # self._norm_factor = 1./self.Xtotal(100, 1)  # could make max mass a
        # parameter

    def process(self, **kwargs):
        """Process module."""
        print(kwargs)
        kroupainput = kwargs['kroupainput']

        self._norm_factor = 1./self.Xtotal(Kroupa, 100, 1)
        mass = self.Mtotal(kroupainput, self._norm_factor)
        return mass

    def X0(self, m, k):
        """Kroupa CDF part 1 from Kroupa 2001b.

        CDF of mass in range 0.01 to
        m solar masses where m < 0.08 & k is
        k = 1/(integral of CDF over full range)
        = normalization
        """
        return (k/0.7 * 0.08**0.3 * (m**0.7 - 0.01**0.7))

    def X1(self, m, k):
        """Kroupa CDF part 2 from Kroupa 2001b.

        mass range 0.08 - 0.5
        """
        return (k/-0.3) * 0.08**1.3 * (m**(-0.3) - 0.08**(-0.3))

    def X2(self, m, k):
        """Kroupa CDF part 3 from Kroupa 2001b.

        mass > 0.5
        """
        return (k/-1.3) * 0.08**1.3 * (0.5 * m**(-1.3) - 0.5**(-0.3))

    def Xtotal(self, m, k):
        """Kroupa full CDF from Kroupa 2001b.

        0.01 <= mass, k is normalization
        """
        if m < 0.01:
            return None
        if m < 0.08:
            return self.X0(m, k)
        if m < 0.5:
            return self.X0(0.08, k) + self.X1(m, k)
        return self.X0(0.08, k) + self.X1(0.5, k) + self.X2(m, k)

    def M0(self, f, k):
        """Kroupa inverted CDF part 1 from Kroupa 2001b.

        mass range 0.01 - 0.08
        k is normalization
        """
        return (f * (0.7)/k * 0.08**(-0.3) + 0.01**0.7)**(1/0.7)

    def M1(self, f, k):
        """Kroupa inverted CDF part 2 from Kroupa 2001b.

        mass range 0.08 - 0.5
        k is normalization
        """
        return ((f - self.X0(0.08, k)) * (-0.3)/k * 0.08**(-1.3) +
                0.08**(-0.3))**(1/-0.3)

    def M2(self, f, k):
        """Kroupa inverted CDF part 2 from Kroupa 2001b.

        mass > 0.5
        k is normalization
        """
        return (((f - self.X1(0.5, k) - self.X0(0.08, k)) * -1.3/k *
                0.5**(-2.3) * (6.25)**1.3 + 0.5**(-1.3))**(1/-1.3))

    def Mtotal(self, f, k):
        """Kroupa full inverted CDF from Kroupa 2001b.

        k is normalization,
        k = 1/(integral of CDF over full mass range)
        """
        if f < self.Xtotal(0.08, k):
            return self.M0(f, k)
        elif f < self.Xtotal(0.5, k):
            return self.M1(f, k)
        return self.M2(f, k)
