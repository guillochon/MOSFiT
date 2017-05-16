"""Definitions for the `LightCurve` class."""
from collections import OrderedDict

import numpy as np

from mosfit.modules.outputs.output import Output


# Important: Only define one ``Module`` class per file.


class LightCurve(Output):
    """Output a light curve to disk."""

    _lc_keys = [
        'magnitudes', 'e_magnitudes', 'model_observations', 'countrates',
        'e_countrates',
        'all_telescopes', 'all_bands', 'all_systems', 'all_instruments',
        'all_bandsets', 'all_modes',
        'all_times', 'all_frequencies', 'observed', 'all_band_indices',
        'observation_types'
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
            omags = output['observation_types'] == 'magnitude'
            output['model_variances'] = np.zeros_like(output[
                'model_observations'])
            output['model_upper_limits'] = np.full(len(output[
                'model_observations']), False)
            varias = 10.0 ** (-self._limiting_magnitude / 2.5)
            mods = 10.0 ** (
                -np.array(output['model_observations'][omags]) / 2.5)
            output['model_observations'][omags] = -2.5 * np.log10(
                varias * np.random.randn(len(omags)) + mods)
            obsas = 10.0 ** (
                -np.array(output['model_observations'][omags]) / 2.5)
            output['model_variances'][omags] = np.abs(-output[
                'model_observations'][omags] - 2.5 * (
                    np.log10(varias + obsas)))
            ul_mask = omags & (obsas < 3.0 * varias)
            output['model_upper_limits'] = ul_mask
            # output['model_observations'][ul_mask] = self._limiting_magnitude
            return output

        # Then, apply GP predictions, if available.
        if (all([x in kwargs
                 for x in ['kmat', 'kfmat', 'koamat', 'kaomat']]) and not
            any([kwargs[x] is None
                 for x in ['kmat', 'kfmat', 'koamat', 'kaomat']])):
            ikmat = np.linalg.inv(
                kwargs['kmat'] + np.diag(kwargs['kdiagonal']))
            kfmatd = np.diagonal(kwargs['kfmat'])
            koamat = kwargs['koamat']
            kaomat = kwargs['kaomat']
            output['model_variances'] = np.sqrt(kfmatd - np.diagonal(np.matmul(
                np.matmul(kaomat, ikmat), koamat)))
        else:
            output['model_variances'] = np.full(
                len(output['model_observations']), kwargs['abandvs'])

        return output
