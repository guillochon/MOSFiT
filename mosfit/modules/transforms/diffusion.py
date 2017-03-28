"""Definitions for the `Diffusion` class."""
from collections import OrderedDict

import numexpr as ne
import numpy as np
from mosfit.constants import C_CGS, DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.transforms.transform import Transform


# Important: Only define one ``Module`` class per file.


class Diffusion(Transform):
    """Photon diffusion transform."""

    N_INT_TIMES = 1000
    DIFF_CONST = 2.0 * M_SUN_CGS / (13.7 * C_CGS * KM_CGS)
    TRAP_CONST = 3.0 * M_SUN_CGS / (FOUR_PI * KM_CGS ** 2)

    def process(self, **kwargs):
        """Process module."""
        super(Diffusion, self).process(**kwargs)
        self._kappa = kwargs['kappa']
        self._kappa_gamma = kwargs['kappagamma']
        self._m_ejecta = kwargs['mejecta']
        self._v_ejecta = kwargs['vejecta']
        self._tau_diff = np.sqrt(self.DIFF_CONST * self._kappa *
                                 self._m_ejecta / self._v_ejecta) / DAY_CGS
        self._trap_coeff = (
            self.TRAP_CONST * self._kappa_gamma * self._m_ejecta /
            (self._v_ejecta ** 2)) / DAY_CGS ** 2
        td2, A = self._tau_diff ** 2, self._trap_coeff  # noqa: F841

        new_lum = []
        evaled = False
        lum_cache = OrderedDict()
        min_te = min(self._dense_times_since_exp)
        for te in self._times_to_process:
            if te <= 0.0:
                new_lum.append(0.0)
                continue
            if te in lum_cache:
                new_lum.append(lum_cache[te])
                continue
            te2 = te ** 2  # noqa: F841
            tb = max(0.0, min_te)
            int_times = np.linspace(tb, te, self.N_INT_TIMES)
            dt = int_times[1] - int_times[0]

            int_lums = np.interp(  # noqa: F841
                int_times, self._dense_times_since_exp,
                self._dense_luminosities)

            if not evaled:
                int_arg = ne.evaluate('2.0 * int_lums * int_times / td2 * '
                                      'exp((int_times**2 - te2) / td2) * '
                                      '(1.0 - exp(-A / te2))')
                evaled = True
            else:
                int_arg = ne.re_evaluate()
            # int_arg = [
            #     2.0 * l * t / td2 *
            #     np.exp((t**2 - te**2) / td2) * (1.0 - np.exp(-A / te**2))
            #     for t, l in zip(int_times, int_lums)
            # ]
            int_arg[np.isnan(int_arg)] = 0.0
            lum_val = np.trapz(int_arg, dx=dt)
            lum_cache[te] = lum_val
            new_lum.append(lum_val)
        return {'dense_luminosities': new_lum}
