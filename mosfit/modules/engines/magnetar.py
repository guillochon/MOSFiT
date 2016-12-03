from math import isnan

import numpy as np
from mosfit.constants import DAY_CGS
from mosfit.modules.engines.engine import Engine

CLASS_NAME = 'Magnetar'


class Magnetar(Engine):
    """Magnetar spin-down engine
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        if 'densetimes' in kwargs:
            self._times = kwargs['densetimes']
        else:
            self._times = kwargs['resttimes']
        self._Pspin = kwargs['Pspin']
        self._Bfield = kwargs['Bfield']
        self._Mns = kwargs['Mns']
        self._thetaPB = kwargs['thetaPB']
        self._rest_t_explosion = kwargs['resttexplosion']

        Ep = 2.6e52 * (self._Mns / 1.4)**(3. / 2.) * self._Pspin**(-2)

        tp = 1.3e5 * self._Bfield**(-2) * self._Pspin**2 * (self._Mns / 1.4)**(
            3. / 2.) * (np.sin(self._thetaPB))**(-2)

        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]

        # print(ts)
        #
        # raise SystemExit
        #
        luminosities = [Ep / tp / (1. + t * DAY_CGS / tp)**2 for t in ts]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        # Add on to any existing luminosity
        luminosities = self.add_to_existing_lums(luminosities)

        return {'luminosities': luminosities}
