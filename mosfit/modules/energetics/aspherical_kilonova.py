"""Definitions for the `AsphericalKilonova` class."""
import numpy as np
from astrocats.catalog.source import SOURCE

from mosfit.modules.energetics.energetic import Energetic


# Important: Only define one ``Module`` class per file.


class AsphericalKilonova(Energetic):
    """
        Uses viewing angle and lanthanide-poor opening angle from Darbha and
        Kasen 2020.

        NOTE only valid for theta_open <= pi/2
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2020ApJ...897..150D'}
    ]

    def process(self, **kwargs):
        """Process module."""
        # Viewing angle
        self._cos_theta = kwargs[self.key('cos_theta')]
        # Opening angle
        self._cos_theta_open = kwargs[self.key('cos_theta_open')]

        # Dharba and Kasen 2020 scalings for conical opening in opaque sphere

        ct = (1-self._cos_theta_open**2)**0.5

        if self._cos_theta > ct:
            Aproj_top = np.pi * ct * self._cos_theta
        else:
            theta_p = np.arccos(self._cos_theta_open /
                                (1 - self._cos_theta**2)**0.5)
            theta_d = np.arctan(np.sin(theta_p) / self._cos_theta_open *
                        (1 - self._cos_theta**2)**0.5 / np.abs(self._cos_theta))
            Aproj_top = (theta_p - np.sin(theta_p)*np.cos(theta_p)) - (ct *
                            self._cos_theta*(theta_d -
                            np.sin(theta_d)*np.cos(theta_d) - np.pi))

        minus_cos_theta = -1 * self._cos_theta

        if minus_cos_theta < -1 * ct:
            Aproj_bot = 0
        else:
            theta_p2 = np.arccos(self._cos_theta_open /
                                (1 - minus_cos_theta**2)**0.5)
            theta_d2 = np.arctan(np.sin(theta_p2) / self._cos_theta_open *
                        (1 - minus_cos_theta**2)**0.5 / np.abs(minus_cos_theta))

            Aproj_bot1 = (theta_p2 - np.sin(theta_p2)*np.cos(theta_p2)) + (ct *
            minus_cos_theta*(theta_d2 - np.sin(theta_d2)*np.cos(theta_d2)))
            Aproj_bot = np.max([Aproj_bot1, 0])

        Aproj = Aproj_top + Aproj_bot


        # Compute reference areas for this opening angle to scale luminosity

        cos_theta_ref = 0.5

        if cos_theta_ref > ct:
            Aref_top = np.pi * ct * cos_theta_ref
        else:
            theta_p_ref = np.arccos(self._cos_theta_open /
                                (1 - cos_theta_ref**2)**0.5)
            theta_d_ref = np.arctan(np.sin(theta_p_ref) / self._cos_theta_open *
                        (1 - cos_theta_ref**2)**0.5 / np.abs(cos_theta_ref))
            Aref_top = (theta_p_ref - np.sin(theta_p_ref) *
                        np.cos(theta_p_ref)) - (ct * cos_theta_ref *
                        (theta_d_ref - np.sin(theta_d_ref) *
                        np.cos(theta_d_ref) - np.pi))

        minus_cos_theta_ref = -1 * cos_theta_ref

        if minus_cos_theta_ref < -1 * ct:
            Aref_bot = 0
        else:
            theta_p2_ref = np.arccos(self._cos_theta_open /
                                (1 - minus_cos_theta_ref**2)**0.5)
            theta_d2_ref = np.arctan(np.sin(theta_p2_ref) /
                    self._cos_theta_open * (1 - minus_cos_theta_ref**2)**0.5 /
                        np.abs(minus_cos_theta_ref))

            Aref_bot = (theta_p2_ref - np.sin(theta_p2_ref) *
                        np.cos(theta_p2_ref)) + (ct * minus_cos_theta_ref *
                        (theta_d2_ref - np.sin(theta_d2_ref) *
                        np.cos(theta_d2_ref)))

        Aref = Aref_top + Aref_bot

        Ablue = Aproj
        Ablue_ref = Aref

        Ared = np.pi - Ablue
        Ared_ref = np.pi - Ablue_ref


        return {self.key('area_blue'): Ablue,
                self.key('area_blue_ref'): Ablue_ref,
                self.key('area_red'): Ared,
                self.key('area_red_ref'): Ared_ref
                }
