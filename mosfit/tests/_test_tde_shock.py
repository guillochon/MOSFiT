"""Shock-efficiency TDE fallback and the ``tde_shock`` model."""
from __future__ import print_function

import os
import sys

import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    from schwimmbad import SerialPool

    from mosfit.constants import C_CGS, M_SUN_CGS
    from mosfit.fitter import Fitter
    from mosfit.model import Model
    from mosfit.modules.engines.fallback import Fallback
    from mosfit.tests.dummies import DummyModel

    import astropy.constants as c

    times = np.unique(np.concatenate((
        [0.0], np.logspace(-6, 2, 100) + 10.0, np.linspace(0, 120, 40))))

    fb = Fallback(name='fallback', model=DummyModel())
    fb._provide_dense = True

    # MOSFiT ``b`` is scaled; physical beta=1 is b=0.32 for gamma=4/3.
    # Spec epsilon numbers use physical beta=1; Lpeak~8e43 used MOSFiT b=1.
    b_beta1 = (1.0 - 0.6) / 1.25
    shared = dict(
        b=b_beta1, starmass=1.0, resttexplosion=10.0, Leddlim=1.0,
        dense_times=times)

    def _expected_shock(frad, mh, mstar, rstar, beta):
        rt = rstar * c.R_sun.cgs.value * (mh / mstar) ** (1.0 / 3.0)
        rp = rt / beta
        rg = c.G.cgs.value * mh * M_SUN_CGS / (C_CGS * C_CGS)
        raw = frad * rg / rp
        return min(raw, 0.42), rp / rg

    out6 = fb.process(frad=0.03, bhmass=1.0e6, **shared)
    np.testing.assert_allclose(out6['beta'], 1.0, rtol=1e-12)
    exp_eff, exp_rprg = _expected_shock(
        0.03, 1.0e6, 1.0, out6['Rstar'], out6['beta'])
    np.testing.assert_allclose(out6['rp_over_rg'], exp_rprg, rtol=1e-12)
    np.testing.assert_allclose(out6['efficiency'], exp_eff, rtol=1e-12)
    # Spec numbers assume R_*=Rsun; Tout ZAMS R(1 Msun) is ~0.9 Rsun.
    np.testing.assert_allclose(out6['rp_over_rg'], 47.0, rtol=0.15)
    np.testing.assert_allclose(out6['efficiency'], 6.4e-4, rtol=0.15)
    print('Mh=1e6 beta=1 efficiency', out6['efficiency'], 'rp_over_rg',
          out6['rp_over_rg'], 'Rstar', out6['Rstar'])

    out_b1 = fb.process(
        frad=0.03, bhmass=1.0e6, b=1.0, starmass=1.0, resttexplosion=10.0,
        Leddlim=1.0, dense_times=times)
    peak_b1 = float(np.max(np.asarray(out_b1['dense_luminosities'])))
    print('Mh=1e6 MOSFiT b=1 beta', out_b1['beta'], 'Lpeak', peak_b1,
          'Ledd', out_b1['Ledd'])
    np.testing.assert_allclose(peak_b1, 8.0e43, rtol=0.25)
    if not (peak_b1 < out_b1['Ledd']):
        raise SystemExit('10^6 shock peak should sit below Ledd')

    out7 = fb.process(frad=0.03, bhmass=1.0e7, **shared)
    np.testing.assert_allclose(out7['beta'], 1.0, rtol=1e-12)
    exp_eff7, exp_rprg7 = _expected_shock(
        0.03, 1.0e7, 1.0, out7['Rstar'], out7['beta'])
    np.testing.assert_allclose(out7['rp_over_rg'], exp_rprg7, rtol=1e-12)
    np.testing.assert_allclose(out7['efficiency'], exp_eff7, rtol=1e-12)
    np.testing.assert_allclose(out7['rp_over_rg'], 10.0, rtol=0.15)
    np.testing.assert_allclose(out7['efficiency'], 3.0e-3, rtol=0.15)
    print('Mh=1e7 efficiency', out7['efficiency'], 'rp_over_rg',
          out7['rp_over_rg'])

    # Efficiency formula uses the gamma-interpolated physical beta.
    low = dict(
        b=0.7, starmass=0.5, bhmass=1.0e6, resttexplosion=10.0,
        Leddlim=1.0, dense_times=times, frad=0.03)
    out_low = fb.process(**low)
    rt = (out_low['Rstar'] * c.R_sun.cgs.value *
          (1.0e6 / 0.5) ** (1.0 / 3.0))
    rp = rt / out_low['beta']
    rg = c.G.cgs.value * 1.0e6 * M_SUN_CGS / (C_CGS * C_CGS)
    np.testing.assert_allclose(
        out_low['efficiency'], min(0.03 * rg / rp, 0.42), rtol=1e-12)
    np.testing.assert_allclose(out_low['rp_over_rg'], rp / rg, rtol=1e-12)
    print('low-mass interpolated beta', out_low['beta'], 'efficiency',
          out_low['efficiency'])

    # Same hydro inputs: dmdt/tpeak/tfallback/beta/Rstar match efficiency mode.
    shock_kw = dict(
        b=0.7, starmass=0.5, bhmass=1e7, resttexplosion=10.0,
        Leddlim=1.0, dense_times=times)
    out_eff = fb.process(efficiency=0.1, **shock_kw)
    out_frad = fb.process(frad=0.03, **shock_kw)
    for key in ('dmdt', 'tpeak', 'tfallback', 'beta', 'Rstar'):
        np.testing.assert_allclose(out_eff[key], out_frad[key], rtol=1e-15)
    if np.array_equal(
            np.asarray(out_eff['dense_luminosities']),
            np.asarray(out_frad['dense_luminosities'])):
        raise SystemExit('luminosities should differ between modes')
    if 'rp_over_rg' in out_eff:
        raise SystemExit('efficiency mode should not set rp_over_rg')
    np.testing.assert_allclose(out_eff['efficiency'], 0.1, rtol=1e-15)
    print('tde vs tde_shock hydro keys match')

    # Existing tde path: golden luminosities unchanged when efficiency is set.
    golden_kw = dict(
        b=0.7, starmass=0.5, bhmass=1e7, resttexplosion=10.0,
        efficiency=0.1, Leddlim=1.0, dense_times=times)
    out_g = fb.process(**golden_kw)
    lum = np.asarray(out_g['dense_luminosities'], dtype=float)
    np.testing.assert_allclose(np.sum(lum), 2.7710499828301767e+46, rtol=1e-10)
    np.testing.assert_allclose(np.max(lum), 7.638370418643626e+44, rtol=1e-10)
    print('efficiency-mode golden luminosities unchanged')

    # Model.run at the canonical shock parameters.
    fitter = Fitter(
        test=True, quiet=True, exit_on_prompt=True, prefer_cache=True)
    dummy = fitter.generate_dummy_data(name='tde_shock')
    pool = SerialPool()
    model = Model(
        model='tde_shock',
        data=dummy,
        test=True,
        printer=fitter._printer,
        fitter=fitter,
        pool=pool)
    ok = model.load_data(dummy, event_name='tde_shock', pool=pool)
    if not ok:
        raise SystemExit('tde_shock load_data failed')
    if 'frad' not in model._free_parameters:
        raise SystemExit('frad is not a free parameter of tde_shock')
    if 'efficiency' in model._free_parameters:
        raise SystemExit('efficiency should not be free in tde_shock')
    if 'Tviscous' in model._call_stack:
        raise SystemExit('Tviscous should not be in tde_shock')
    if 'viscous' in model._call_stack:
        raise SystemExit('viscous transform should not be in tde_shock')

    targets = {
        'frad': 0.03,
        'starmass': 1.0,
        'b': b_beta1,
        'bhmass': 1.0e6,
        'texplosion': -12.0,
        'lphoto': 1.5,
        'Rph0': 6.31,
        'nhhost': 1.0e21,
        'variance': 1.0e-2,
    }
    x = []
    for name in model._free_parameters:
        par = model._modules[name]
        if name in targets and par._min_value is not None:
            x.append(float(par.fraction(targets[name])))
        else:
            x.append(0.5)
    x = np.asarray(x, dtype=float)
    outputs = model.run(x)
    if 'efficiency' not in outputs:
        raise SystemExit('Model.run did not return derived efficiency')
    if 'rp_over_rg' not in outputs:
        raise SystemExit('Model.run did not return rp_over_rg')
    np.testing.assert_allclose(outputs['efficiency'], 6.4e-4, rtol=0.15)
    np.testing.assert_allclose(outputs['rp_over_rg'], 47.0, rtol=0.15)
    print('tde_shock Model.run efficiency', outputs['efficiency'],
          'rp_over_rg', outputs['rp_over_rg'],
          'ndim', model._num_free_parameters)
    print('tde_shock checks passed')
    sys.exit(0)
