"""Definitions for the `HomologousExpansion` class."""
import numpy as np

from mosfit.constants import FOE, KM_CGS, M_SUN_CGS
from mosfit.modules.energetics.energetic import Energetic


# Important: Only define one ``Module`` class per file.


class HomologousExpansion(Energetic):
    """Generate `vejecta` from `kinetic_energy` assuming homologous expansion.

    Convert an input kinetic energy to velocity assuming ejecta in
    homologous expansion.
    """

    def process(self, **kwargs):
        """Process module."""
        self._energy = kwargs[self.key('kinetic_energy')]
        self._m_ejecta = kwargs[self.key('mejecta')]

        v_ejecta = np.sqrt(10 * self._energy * FOE /
                           (3 * self._m_ejecta * M_SUN_CGS)) / KM_CGS

        return {self.key('vejecta'): v_ejecta}
