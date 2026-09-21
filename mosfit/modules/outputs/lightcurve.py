"""Definitions for the `LightCurve` class."""
from collections import OrderedDict

import numpy as np

from mosfit.modules.outputs.output import Output


# Important: Only define one ``Module`` class per file.


class LightCurve(Output):
    """Output a light curve to disk."""

    _lc_keys = [
        'magnitudes', 'e_magnitudes', 'model_observations', 'countrates',
        'e_countrates', 'all_telescopes', 'all_bands', 'all_systems',
        'all_instruments', 'all_bandsets', 'all_modes', 'all_times',
        'all_frequencies', 'observed', 'all_band_indices', 'observation_types'
    ]

    def __init__(self, **kwargs):
        """Initialize module."""
        super(LightCurve, self).__init__(**kwargs)
        self._dense_keys = self._lc_keys
        self._limiting_magnitude = self._model._fitter._limiting_magnitude

    def process(self, **kwargs):
        """Process module."""
        # First, rename some keys.
        output = OrderedDict()
        for key in sorted(kwargs.keys()):
            if key in self._dense_keys:
                continue
            output[key] = kwargs[key]
        for key in self._dense_keys:
            output[key.replace('all_', '')] = kwargs[key]

        if self._limiting_magnitude is not None:
            ls = 0.0
            if isinstance(self._limiting_magnitude, list):
                lm = float(self._limiting_magnitude[0])
                if len(self._limiting_magnitude) > 1:
                    ls = float(self._limiting_magnitude[1])
            else:
                lm = self._limiting_magnitude

            obs = np.array(output['model_observations'], dtype=float)
            lmo = len(obs)

            omags = np.array(
                [x == 'magnitude' for x in output['observation_types']],
                dtype=bool)
            n_mags = int(np.count_nonzero(omags))
            output['model_variances'] = np.zeros(lmo, dtype=float)
            output['model_upper_limits'] = np.full(lmo, False)
            lms = lm + ls * np.random.randn(lmo)
            varias = 10.0 ** (-lms / 2.5)

            # Scatter each magnitude row in flux space by the flux error the
            # limiting magnitude implies. Every array indexed here is masked
            # the same way; mixing masked and full-length arrays silently
            # broadcasts (or raises) once a model has non-magnitude rows.
            mods = 10.0 ** (-obs[omags] / 2.5)
            drawn = varias[omags] * np.random.randn(n_mags) + mods
            obsas = np.zeros(lmo, dtype=float)
            obsas[omags] = drawn
            with np.errstate(divide='ignore', invalid='ignore'):
                obs[omags] = -2.5 * np.log10(drawn)
                output['model_variances'][omags] = np.abs(
                    -obs[omags] - 2.5 * np.log10(varias[omags] + drawn))

            # A draw at or below zero flux is a non-detection, not a missing
            # observation; treat it as one rather than leaving a NaN that
            # would quietly drop the epoch from the mock light curve.
            ul_mask = omags & ((obsas < 3.0 * varias) | ~np.isfinite(obs))
            output['model_upper_limits'] = ul_mask
            obs[ul_mask] = lms[ul_mask]
            output['model_variances'][ul_mask] = 2.5 * (
                np.log10(2.0 * varias[ul_mask]) - np.log10(varias[ul_mask]))
            output['model_observations'] = obs
            return output

        output['model_variances'] = np.full(
            len(output['model_observations']), kwargs['abandvs'])

        return output
