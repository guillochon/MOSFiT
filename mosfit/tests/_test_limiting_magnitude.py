"""Mock-survey noise (`--limiting-magnitude`) with mixed observation types.

Regression: the magnitude branch mixed boolean-masked and full-length arrays,
which only worked while every row of a generative run happened to be a
magnitude. A model producing a bolometric luminosity or a radio flux density
alongside magnitudes raised `ValueError`.
"""
from __future__ import print_function

import os
import sys

import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

    from mosfit.modules.outputs.lightcurve import LightCurve
    from mosfit.tests.dummies import DummyModel

    dummy = DummyModel()
    dummy._fitter._limiting_magnitude = [23.0, 0.5]
    lc = LightCurve(name='fitlc', model=dummy)
    nobs = 6
    types = np.array(
        ['magnitude', 'luminosity', 'magnitude', 'fluxdensity', 'magnitude',
         'magnitude'], dtype=object)
    lc_out = lc.process(
        magnitudes=np.zeros(nobs),
        e_magnitudes=np.zeros(nobs),
        model_observations=np.full(nobs, 22.0),
        countrates=np.zeros(nobs),
        e_countrates=np.zeros(nobs),
        all_telescopes=[''] * nobs,
        all_bands=['V'] * nobs,
        all_systems=[''] * nobs,
        all_instruments=[''] * nobs,
        all_bandsets=[''] * nobs,
        all_modes=[''] * nobs,
        all_times=np.arange(nobs, dtype=float),
        all_frequencies=np.zeros(nobs),
        observed=np.ones(nobs, dtype=bool),
        all_band_indices=np.zeros(nobs, dtype=int),
        observation_types=types,
        abandvs=0.1)
    obs = np.asarray(lc_out['model_observations'], dtype=float)
    assert obs.shape == (nobs,)
    mag_rows = types == 'magnitude'
    # Non-magnitude rows are untouched, and no magnitude row is left as NaN:
    # a draw below zero flux is a non-detection, not a dropped epoch.
    assert np.all(obs[~mag_rows] == 22.0)
    assert np.all(np.isfinite(obs[mag_rows]))
    assert not np.any(lc_out['model_upper_limits'][~mag_rows])
    assert np.all(np.isfinite(lc_out['model_variances']))

    print('Limiting magnitude noise tests passed.')
    sys.exit(0)
