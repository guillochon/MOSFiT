"""Tests for the rest-frame SED interface and the mock-survey noise model."""
from __future__ import print_function

import os

import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

    from mosfit.fitter import Fitter
    from mosfit.lynx import (
        DEFAULT_WAVELENGTH_GRID, LYNX_FIXED_PARAMETERS, LynxSource,
        lynx_wavelength_grid)
    from mosfit.model import Model

    # --- Wavelength grid spec -------------------------------------------
    grid = lynx_wavelength_grid()
    assert grid[0] == DEFAULT_WAVELENGTH_GRID[0]
    assert grid[-1] == DEFAULT_WAVELENGTH_GRID[1]
    assert len(grid) == DEFAULT_WAVELENGTH_GRID[2]
    assert np.array_equal(
        lynx_wavelength_grid([100.0, 200.0, 3]), np.array([100., 150., 200.]))
    for bad in ([100.0, 50.0, 10], [0.0, 10.0, 10], [1.0, 2.0, 1], [1.0, 2.0]):
        try:
            lynx_wavelength_grid(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted bad grid spec {}'.format(bad))

    # --- Rest-frame SED source ------------------------------------------
    phases = np.array([10.0, 20.0, 40.0])
    wavelengths = np.linspace(3000., 9000., 40)
    source = LynxSource(
        model='default', phases=phases, wavelengths=wavelengths)

    free = source.free_parameter_names()
    assert free, 'model reported no free parameters'
    # Lynx owns these, so none of them may vary underneath it.
    assert not set(free) & set(LYNX_FIXED_PARAMETERS)

    sed = source.compute_sed()
    assert sed.shape == (phases.size, wavelengths.size), sed.shape
    assert np.all(np.isfinite(sed))
    assert np.any(sed > 0.0)

    assert source.minwave() == 3000.0 and source.maxwave() == 9000.0
    assert source.minphase() is not None and source.maxphase() is not None
    assert source.minphase() <= phases.min()
    assert source.maxphase() >= phases.max()

    # Explicit fractions and midpoint parameters agree.
    mid = np.full(len(free), 0.5)
    assert np.allclose(sed, source.compute_sed(fractions=mid))

    # Physical values map onto the unit cube and back.
    manifest = {x['name']: x for x in source.parameter_manifest()}
    name = free[0]
    entry = manifest[name]
    assert entry['free'] and entry['index'] == 0
    target = 0.5 * (entry['min_value'] + entry['max_value'])
    fracs = source.fractions_from_parameters({name: target})
    assert 0.0 <= fracs[0] <= 1.0
    try:
        source.fractions_from_parameters({'not_a_parameter': 1.0})
    except ValueError:
        pass
    else:
        raise AssertionError('accepted an unknown parameter name')

    # Fixed parameters carry their pinned values into the manifest.
    assert manifest['redshift']['value'] == 0.0
    assert not manifest['redshift']['free']

    # --- Flux units --------------------------------------------------------
    # The SED is nJy at 10 pc, so a monochromatic AB magnitude taken from it
    # must match the absolute magnitude MOSFiT's own photometry reports for
    # the same parameters at the same distance.
    fitter = Fitter(quiet=True)
    data = fitter.generate_dummy_data(
        'default', max_time=float(phases.max()), time_list=list(phases),
        band_list=['V'])
    model = Model(
        model='default', data=data, fitter=fitter, printer=fitter._printer)
    fixed = []
    for key, value in LYNX_FIXED_PARAMETERS.items():
        fixed += [key, value]
    model.load_data(
        data, event_name='default', time_list=list(phases), band_list=['V'],
        user_fixed_parameters=fixed)
    out = model.run_stack(mid, root='output')
    times = np.asarray(out['times'], dtype=float)
    mags = np.asarray(out['model_observations'], dtype=float)

    iwav = int(np.argmin(np.abs(wavelengths - 5500.0)))
    ab_from_sed = -2.5 * np.log10(sed[:, iwav] / 3.631e12)
    for pi, phase in enumerate(phases):
        ti = int(np.argmin(np.abs(times - phase)))
        # Monochromatic vs band-integrated, so allow a modest tolerance.
        assert abs(ab_from_sed[pi] - mags[ti]) < 0.25, (
            phase, ab_from_sed[pi], mags[ti])

    # Parameter manifest and phase range are also reachable from the model.
    assert model.minwave() is not None and model.maxwave() is not None
    assert [x['name'] for x in model.parameter_manifest(include_fixed=False)]

    print('Lynx interface tests passed.')
