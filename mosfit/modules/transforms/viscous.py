"""Definitions for the `Viscous` class."""
import numexpr as ne
import numpy as np
from mosfit.modules.transforms.transform import Transform
from scipy.interpolate import interp1d

CLASS_NAME = 'Viscous'


class Viscous(Transform):
    """Viscous delay transform."""

    logsteps = True
    N_INT_TIMES = 1000
    testnum = 0

    def process(self, **kwargs):
        """Process module."""
        super(Viscous, self).process(**kwargs)

        ipeak = np.argmax(self._dense_luminosities)
        tpeak = self._dense_times_since_exp[ipeak]
        Tvisc = kwargs['Tviscous'] * tpeak

        new_lum = []
        evaled = False
        lum_cache = {}
        lum_func = interp1d(self._dense_times_since_exp,
                            self._dense_luminosities)
        timesteps = self.N_INT_TIMES
        min_te = min(self._dense_times_since_exp)

        for j, te in enumerate(self._times_to_process):
            if te <= 0.0:
                new_lum.append(0.0)
                continue
            if lum_func(te) <= 0:
                new_lum.append(0.0)
                continue

            if te in lum_cache:
                new_lum.append(lum_cache[te])
                continue

            if self.logsteps:
                min_te = 1.0e-4
                int_times = np.logspace(
                    np.log10(min_te), np.log10(te), num=self.N_INT_TIMES)
                if int_times[0] < min_te:
                    int_times[0] = min_te
                if int_times[-1] > te:
                    int_times[-1] = te
            else:
                int_times = np.linspace(min_te, te, timesteps)
                # all the spacings are the same bc of linspace
                dt = int_times[1] - int_times[0]

            int_lums = lum_func(int_times)

            if not evaled:
                int_arg = ne.evaluate(
                    'exp((-te + int_times)/Tvisc) * int_lums')
                evaled = True
            else:
                int_arg = ne.re_evaluate()

            # could also make cuts on luminosity here
            int_arg[np.isnan(int_arg)] = 0.0
            # print (int_arg)]

            if self.logsteps:
                lum_val = np.trapz(int_arg, int_times) / Tvisc
            else:
                lum_val = np.trapz(int_arg, dx=dt) / Tvisc

            lum_cache[te] = lum_val
            new_lum.append(lum_val)

        return {'dense_luminosities': new_lum}
