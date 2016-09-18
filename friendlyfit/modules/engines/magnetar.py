from math import isnan

import numpy as np

from ..module import Module

CLASS_NAME = 'Magnetar'


class Magnetar(Module):
    """Magnetar spin-down engine
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        if 'densetimes' in kwargs:
            self._times = kwargs['densetimes']
        else:
            self._times = kwargs['times']
        self._Pspin = kwargs['Pspin']
        self._Bfield = kwargs['Bfield']
        self._Mns = kwargs['Mns']
        self._thetaPB = kwargs['thetaPB']
        self._texplosion = kwargs['texplosion']

        Ep = 2.6e52 * (self._Mns / 1.4)**(3. / 2.) * self._Pspin**(-2)

        tp = 1.3e5 * self._Bfield**(-2) * self._Pspin**2 * (self._Mns / 1.4)**(
            3. / 2.) * (np.sin(self._thetaPB))**(-2)

        ts = [np.inf if self._texplosion > x else (x - self._texplosion)
              for x in self._times]

        luminosities = [Ep / tp / (1. + t / tp)**2 for t in ts]
        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        # Add on to any existing luminosity
        old_luminosities = kwargs.get('luminosities', None)
        if old_luminosities is not None:
            luminosities = [x + y
                            for x, y in zip(old_luminosities, luminosities)]

        return {'luminosities': luminosities}
