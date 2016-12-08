import numpy as np
from mosfit.constants import FOE, M_SUN_CGS, KM_CGS
from mosfit.modules.energetics.energetic import energetic

CLASS_NAME = 'homologous_expansion'


class homologous_expansion(energetic):
    """Convert an input kinetic energy to velocity assuming ejecta in
    homologous expansion
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        self._energy = kwargs['kinetic_energy']
        self._m_ejecta = kwargs['mejecta']

        v_ejecta = np.sqrt(10 * self._energy * FOE /
                            (3 *self._m_ejecta * M_SUN_CGS)) / KM_CGS

        return {'vejecta': v_ejecta}
