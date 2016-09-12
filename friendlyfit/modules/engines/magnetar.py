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
        self._times = kwargs['times']
        self._Pspin = kwargs['Pspin']
        self._Bfield = kwargs['Bfield']
        self._Mns = kwargs['Mns']
        self._thetaPB = kwargs['thetaPB']
        self._texplosion = kwargs['texplosion']

        Ep = 2.6e52*(Mns/1.4)**(3./2.)*Pspin**(-2)

        tp = 1.3e5*Bfield**(-2)*Pspin**2*(Mns/1.4)**(3./2.)*(np.sin(thetaPB))**(-2)

        ts = [np.inf if self._texplosion > x else (x - self._texplosion)
              for x in self._times]
              
        luminosities = [Ep/tp/(1.+ts/tp)**2
                        for t in ts]

#        print(max(luminosities))
        return {'luminosities': luminosities}
