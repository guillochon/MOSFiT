import numpy as np
from mosfit.constants import FOE, M_SUN_CGS, KM_CGS
from mosfit.modules.energetics.energetic import energetic

CLASS_NAME = 'thin_shell'


class thin_shell(energetic):
    """Convert an input kinetic energy to velocity assuming ejecta in thin shell
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._energy = kwargs['kinetic_energy']
        self._m_ejecta = kwargs['mejecta']

        v_ejecta = np.sqrt(2 * self._energy * FOE /
                            (self._m_ejecta * M_SUN_CGS)) / KM_CGS

        return {'vejecta': v_ejecta}
