"""Definitions for the `LynxSED` class."""
from collections import OrderedDict

import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.constants import BOL_BAND_INDEX, FOUR_PI, NJY_CGS, TEN_PC_CGS
from mosfit.modules.outputs.output import Output

# Important: Only define one ``Module`` class per file.


class LynxSED(Output):
    """Emit the rest-frame SED on a caller-supplied grid.

    MOSFiT normally finishes a model by redshifting the SED, reddening it and
    integrating it through bandpasses. External light-curve simulators such as
    `LightCurveLynx <https://lightcurvelynx.readthedocs.io>`_ do all of that
    themselves, and ask a source model only for rest-frame flux density as a
    function of phase and wavelength. This module produces exactly that view,
    and is inert unless ``--lynx`` is set.

    The convention matches ``SEDModel.compute_sed``: a ``(n_phase, n_wave)``
    array of flux densities in nJy, as the source would be seen at 10 pc with
    no redshift, no time dilation and no extinction applied.
    """

    C_OVER_ANG = (c.c / u.Angstrom).cgs.value

    # erg/s/Angstrom at 10 pc -> nJy, modulo the lambda^2 / c factor.
    FLUX_CONST = 1.0 / (FOUR_PI * TEN_PC_CGS ** 2) / NJY_CGS

    def process(self, **kwargs):
        """Process module."""
        if not getattr(self._model._fitter, '_lynx', False):
            return {}

        seds = kwargs.get(self.key('seds'))
        if seds is None:
            return {}
        seds = np.asarray(seds, dtype=float)
        if seds.ndim == 1:
            seds = seds.reshape(seds.shape[0], 1)

        wavs = np.asarray(kwargs['sample_wavelengths'], dtype=float)
        # Every row shares one grid in this mode (see ``SED.receive_requests``).
        wavs = wavs[0] if wavs.ndim > 1 else wavs

        band_indices = np.asarray(kwargs['all_band_indices'])
        rest_times = np.asarray(kwargs['rest_times'], dtype=float)
        rest_t_explosion = float(kwargs[self.key('resttexplosion')])

        # Drop bolometric rows; they carry no wavelength axis.
        keep = band_indices != BOL_BAND_INDEX
        if not np.any(keep):
            return {}
        rows = np.flatnonzero(keep)

        # One row per phase. With a single pseudo-band this is already true,
        # but collapse duplicates so extra bands cannot silently multiply the
        # number of rows the caller gets back.
        phases = rest_times[rows] - rest_t_explosion
        phases, first = np.unique(phases, return_index=True)
        rows = rows[first]

        block = seds[rows]
        if block.shape[1] != wavs.size:
            n = min(block.shape[1], wavs.size)
            fixed = np.zeros((block.shape[0], wavs.size), dtype=float)
            fixed[:, :n] = block[:, :n]
            block = fixed

        # L_lambda [erg/s/Ang] -> F_nu [nJy] at 10 pc.
        fluxes = block * (wavs ** 2 / self.C_OVER_ANG) * self.FLUX_CONST
        fluxes[~np.isfinite(fluxes)] = 0.0

        return OrderedDict([
            ('lynx_phases', phases),
            ('lynx_wavelengths', wavs),
            ('lynx_seds', fluxes),
        ])
