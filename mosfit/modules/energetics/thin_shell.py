"""Definitions for the `ThinShell` class."""
import numpy as np

from mosfit.constants import FOE, KM_CGS, M_SUN_CGS
from mosfit.modules.energetics.energetic import Energetic

# Important: Only define one `Module` class per file.


class ThinShell(Energetic):
    """Convert an input kinetic energy to velocity assuming ejecta in thin shell
    """

    def process(self, **kwargs):
        """Process module."""
        self._energy = kwargs['kinetic_energy']
        self._m_ejecta = kwargs['mejecta']

        v_ejecta = np.sqrt(2 * self._energy * FOE /
                           (self._m_ejecta * M_SUN_CGS)) / KM_CGS

        return {'vejecta': v_ejecta}
