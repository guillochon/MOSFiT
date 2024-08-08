"""Definitions for the `BNSMagnetar` class."""
from math import isnan

import numpy as np
from scipy.integrate import cumulative_trapezoid
from astrocats.catalog.source import SOURCE

from mosfit.constants import DAY_CGS, M_SUN_CGS, KM_CGS, G_CGS, C_CGS
from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class BNSMagnetar(Engine):
    """Magnetar spin-down engine."""

    _REFERENCES = [
        {SOURCE.BIBCODE: '2023arXiv230911340S'},
        {SOURCE.BIBCODE: '1971ApJ...164L..95O'},
        {SOURCE.BIBCODE: '2019LRR....23....1M'},
        {SOURCE.BIBCODE: '2017ApJ...850L..19M'},
        {SOURCE.BIBCODE: '2005ApJ...629..979L'}
    ]

    def process(self, **kwargs):
        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        self._rest_t_explosion = kwargs[self.key('resttexplosion')]
                
        self._Pspin = kwargs[self.key('Pspin')]
        self._Bfield = kwargs[self.key('Bfield')]
        self._epsilon_therm = kwargs[self.key('epsilon_therm')]
        self._albedo = kwargs[self.key('albedo')]
        self._multipicity = kwargs[self.key('pair_multiplicity')]

        self._m_tov = kwargs[self.key('Mtov')]
        self._radius_ns = kwargs[self.key('radius_ns')]
        
        self._Mns = kwargs[self.key('M_rem')]
        
        self._kappa = kwargs[self.key('kappa')]
        self._v_ejecta = kwargs[self.key('vejecta')]

        
        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]
        
#        np.savetxt('t_test.txt',np.array(ts))


        # Moment of inertia, normalised to Lattimer and Schutz 2005
        I_LS = 1.3e45 * (self._Mns / 1.4) ** (3. / 2.)
        
        I = 2./5. * self._Mns * M_SUN_CGS * (self._radius_ns * KM_CGS)**2

    
        # Find extractable energy (Margalit and Metzger 2017)
        
        # Collapse if omega^2 r + Fnuc < G M / r^2
        #     -> omega_crit = sqrt(G (M-Mtov) / r^3)

        if self._Mns > self._m_tov:
            omega_crit = np.sqrt(G_CGS * (self._Mns - self._m_tov) *
                        M_SUN_CGS / (self._radius_ns * KM_CGS)**3)
            P_crit = 2 * np.pi / omega_crit * 1000
        else:
            omega_crit = 0.0
            P_crit = np.inf
            
        
        # Metzger 2019 paramaterise rotational energy
        Ep = 1.0e53 * (I/I_LS) * (self._Mns /
                2.3)**(3. / 2.) * (self._Pspin / 0.7) ** (-2)

        L0 = 7.0e48 * (I/I_LS) * self._Bfield**2 * (self._Pspin /
            0.7) ** (-4) * (self._Mns / 2.3)**(3. / 2.) * (self._radius_ns / 12)**2

        tp = Ep / L0


        E_collapse = 1.0e53 * (I/I_LS) * (self._Mns /
                    2.3)**(3./2.) * (P_crit / 0.7) ** (-2)

        E_available = Ep - E_collapse


        luminosities = [0.0 for t in ts]
        
        t_collapse = 0.0
        
        t_life_over_t = [0.0 for t in ts]

        # If minimimum rotation period to avoid collapse < break-up period, do nothing
        if P_crit > 0.7:

            luminosities = [L0 / (
                1. + t * DAY_CGS / tp) ** 2 for t in ts]
            # ^ From Ostriker and Gunn 1971 eq 4
            luminosities = [0.0 if isnan(x) else x for x in luminosities]


            # Magnetar input stops when extractable energy used up:
                
            E_rad = cumulative_trapezoid(luminosities, x = np.array(ts) * DAY_CGS, initial=0)

            luminosities = [0.0 if E_rad[i]>E_available else luminosities[i] for
                                i in range(len(luminosities))]
            
#            np.savetxt('lums_test.txt',np.array(luminosities))
                        
            # Pair suppression and thermalisation efficiency (Metzger 2019):
                        
            t_life_over_t_0 = 0.6 / (1 - self._albedo) * (self._multipicity /
                    0.1)**0.5 * (self._v_ejecta * KM_CGS / (0.3*C_CGS))**0.5
                
            
            t_life_over_t = [
                    t_life_over_t_0 * (luminosities[i] /
                    1.e45)**0.5 * ts[i]**-0.5 for i in range(len(ts))
                ]
            
#            np.savetxt('tlife_test.txt',np.array(t_life_over_t))

            
            luminosities = [ luminosities[i] * self._epsilon_therm / (1 +
                               t_life_over_t[i]) for i in range(len(luminosities))
                            ]

#            np.savetxt('new_lums_test.txt',np.array(luminosities))


            # Timespan of rotational support:
            supported = np.array(ts)[E_rad < E_available]

            t_collapse = np.nan

            if len(supported) == 0:
                t_collapse = 0.0
                
            elif len(supported) < len(ts):
                t_collapse = max(supported)
                
            else:
                t_collapse = tp / DAY_CGS
                

        return {self.dense_key('luminosities'): luminosities,
#                self.dense_key('luminosities_sd'): luminosities,
                self.key('t_collapse'): t_collapse,
                self.key('E_collapse'): E_collapse,
                self.key('t_mag'): tp / DAY_CGS,
                self.key('E_mag'): Ep,
                self.dense_key('t_life_over_t'): t_life_over_t
                }
