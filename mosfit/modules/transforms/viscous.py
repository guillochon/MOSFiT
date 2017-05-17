"""Definitions for the `Viscous` class."""
import numexpr as ne
import numpy as np
from mosfit.modules.transforms.transform import Transform
from scipy.interpolate import interp1d

CLASS_NAME = 'Viscous'


class Viscous(Transform):
    """Viscous delay transform."""

    logsteps = False
    N_INT_TIMES = 1000  # 1e5

    def process(self, **kwargs):
        """Process module."""
        super(Viscous, self).process(**kwargs)

        ipeak = np.argmax(self._dense_luminosities)
        tpeak = self._dense_times_since_exp[ipeak]
        Tvisc = kwargs['Tviscous'] * tpeak
        # if kwargs['Tviscous'] > 1.0:
        #     self.N_INT_TIMES = 1000
        #     dense_frac = 0.2
        if kwargs['Tviscous'] <= 1:
            extra_steps = int(1000. - 1000*np.log10(kwargs['Tviscous']))
        else:
            extra_steps = int(1000./(1 + np.log10(kwargs['Tviscous'])))

        new_lum = []
        evaled = False
        lum_cache = {}
        lum_func = interp1d(self._dense_times_since_exp,
                            self._dense_luminosities,
                            copy=False, assume_sorted=True)
        min_te = min(self._dense_times_since_exp)
        max_te = max(self._dense_times_since_exp)

        for j, te in enumerate(self._dense_times_since_exp):
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
                base_times = np.linspace(min_te, te, self.N_INT_TIMES)
                if (te - min_te) < 0.01*(max_te - min_te):
                    int_times = np.unique(np.concatenate(
                        (np.linspace(min_te, te, extra_steps), base_times)))
                else:
                    tearly = min_te + 0.005*(max_te - min_te)
                    tm = te - 0.005*(max_te - min_te)
                    int_times = np.unique(np.concatenate(
                        (np.linspace(min_te, tearly, extra_steps), base_times,
                            np.linspace(tm, te, extra_steps))))

                # if te - min_te > 0.005 * (max_te - min_te):
                #    tm = te - 0.005 * (max_te - min_te)
                #    num_sparse = int(self.N_INT_TIMES*0.2)
                #    sparse_times = np.linspace(min_te, tm, num_sparse)
                #    times_nearte = np.linspace(tm, te, self.N_INT_TIMES -
                #                               num_sparse)
                #    int_times = np.concatenate((sparse_times, times_nearte))
                # else:
                #    int_times = np.linspace(min_te, te, self.N_INT_TIMES)
                # all the spacings are the same bc of linspace
                # dt = int_times[1] - int_times[0]

            int_lums = lum_func(int_times)

            # int_arg = np.exp((-te + int_times)/Tvisc) * int_lums
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
                # STOPPED HERE
                lum_val = np.trapz(int_arg, int_times) / Tvisc
                #lum_val = np.trapz(int_arg, dx=dt) / Tvisc
                #lum_val = (np.sum(int_arg[1:-1]) +
                #           0.5 * (int_arg[0] + int_arg[-1]))/Tvisc
                #lum_val *= (int_times[-1] -
                #            int_times[0])/(2.0 * (len(int_times) - 1))

            lum_cache[te] = lum_val
            new_lum.append(lum_val)

            previscous_lums = self._dense_luminosities
            postviscous_lums = new_lum
        return {'dense_luminosities': new_lum,
                'previscous_lums': previscous_lums,
                'postviscous_lums': postviscous_lums,
                'viscous_times': self._dense_times_since_exp}
