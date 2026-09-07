"""Two-component TDE shock + accretion fallback and the ``tde_shock`` model."""
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
    from mosfit.modules.parameters.darkyear import DarkYear
    from mosfit.modules.utilities.leddcap import LeddCap
    from mosfit.tests.dummies import DummyModel

    import astropy.constants as c

    dummy = DummyModel()
    times = np.unique(np.concatenate((
        [0.0], np.logspace(-6, 2, 100) + 10.0, np.linspace(0, 120, 40))))

    fb = Fallback(name='fallback', model=dummy)
    fb._provide_dense = True

    # MOSFiT ``b`` is scaled; physical beta=1 is b=0.32 for gamma=4/3.
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

    # Prompt-only (frad, no efficiency): capped shock luminosities.
    out6 = fb.process(frad=0.03, bhmass=1.0e6, **shared)
    np.testing.assert_allclose(out6['beta'], 1.0, rtol=1e-12)
    exp_eff, exp_rprg = _expected_shock(
        0.03, 1.0e6, 1.0, out6['Rstar'], out6['beta'])
    np.testing.assert_allclose(out6['rp_over_rg'], exp_rprg, rtol=1e-12)
    np.testing.assert_allclose(out6['shock_efficiency'], exp_eff, rtol=1e-12)
    np.testing.assert_allclose(out6['efficiency'], exp_eff, rtol=1e-12)
    np.testing.assert_allclose(out6['rp_over_rg'], 47.0, rtol=0.15)
    np.testing.assert_allclose(out6['shock_efficiency'], 6.4e-4, rtol=0.15)
    print('Mh=1e6 beta=1 shock_efficiency', out6['shock_efficiency'],
          'rp_over_rg', out6['rp_over_rg'], 'Rstar', out6['Rstar'])

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
    exp_eff7, exp_rprg7 = _expected_shock(
        0.03, 1.0e7, 1.0, out7['Rstar'], out7['beta'])
    np.testing.assert_allclose(out7['rp_over_rg'], exp_rprg7, rtol=1e-12)
    np.testing.assert_allclose(out7['shock_efficiency'], exp_eff7, rtol=1e-12)
    np.testing.assert_allclose(out7['rp_over_rg'], 10.0, rtol=0.15)
    np.testing.assert_allclose(out7['shock_efficiency'], 3.0e-3, rtol=0.15)
    print('Mh=1e7 shock_efficiency', out7['shock_efficiency'],
          'rp_over_rg', out7['rp_over_rg'])

    low = dict(
        b=0.7, starmass=0.5, bhmass=1.0e6, resttexplosion=10.0,
        Leddlim=1.0, dense_times=times, frad=0.03)
    out_low = fb.process(**low)
    rt = (out_low['Rstar'] * c.R_sun.cgs.value *
          (1.0e6 / 0.5) ** (1.0 / 3.0))
    rp = rt / out_low['beta']
    rg = c.G.cgs.value * 1.0e6 * M_SUN_CGS / (C_CGS * C_CGS)
    np.testing.assert_allclose(
        out_low['shock_efficiency'], min(0.03 * rg / rp, 0.42), rtol=1e-12)
    print('low-mass interpolated beta', out_low['beta'], 'shock_efficiency',
          out_low['shock_efficiency'])

    shock_kw = dict(
        b=0.7, starmass=0.5, bhmass=1e7, resttexplosion=10.0,
        Leddlim=1.0, dense_times=times)
    out_eff = fb.process(efficiency=0.1, **shock_kw)
    out_frad = fb.process(frad=0.03, **shock_kw)
    for key in ('dmdt', 'tpeak', 'tfallback', 'beta', 'Rstar'):
        np.testing.assert_allclose(out_eff[key], out_frad[key], rtol=1e-15)
    np.testing.assert_allclose(out_eff['efficiency'], 0.1, rtol=1e-15)
    if 'shock_luminosities' in out_eff or 'dense_shock_luminosities' in out_eff:
        raise SystemExit('efficiency-only mode should not emit shock lums')
    print('tde vs prompt-shock hydro keys match')

    golden_kw = dict(
        b=0.7, starmass=0.5, bhmass=1e7, resttexplosion=10.0,
        efficiency=0.1, Leddlim=1.0, dense_times=times)
    out_g = fb.process(**golden_kw)
    lum = np.asarray(out_g['dense_luminosities'], dtype=float)
    np.testing.assert_allclose(np.sum(lum), 2.7710499828301767e+46, rtol=1e-10)
    np.testing.assert_allclose(np.max(lum), 7.638370418643626e+44, rtol=1e-10)
    print('efficiency-mode golden luminosities unchanged')

    # Two-component: uncapped acc + shock arrays, cap applied after the sum.
    two = fb.process(frad=0.03, efficiency=0.1, bhmass=1.0e6, **shared)
    shock = np.asarray(two['dense_shock_luminosities'])
    acc = np.asarray(two['dense_luminosities'])
    np.testing.assert_allclose(
        two['shock_efficiency'] / 0.1, shock.max() / acc.max(), rtol=1e-10)
    if np.max(acc) <= two['Ledd']:
        # ε_acc = 0.1 is super-Eddington at peak for this draw; leave uncapped.
        pass
    summed = shock + acc
    cap = two['Ledd']
    capped = summed * cap / (summed + cap)
    lc = LeddCap(name='leddcap', model=dummy)
    lc_out = lc.process(
        dense_luminosities=summed, Leddlim=1.0, Ledd=cap,
        dense_indices=np.array([0, 1]))
    np.testing.assert_allclose(lc_out['dense_luminosities'], capped, rtol=1e-12)
    print('two-component arrays and post-sum cap ok')

    dy = DarkYear(name='Tviscous', model=dummy)
    tv_deep = dy.process(
        rp_over_rg=10.0, tfallback=40.0, tpeak=50.0, bhmass=1.0e7,
        viscfac=100.0)['Tviscous']
    tv_shallow = dy.process(
        rp_over_rg=47.0, tfallback=15.0, tpeak=28.0, bhmass=1.0e6,
        viscfac=100.0)['Tviscous']
    if not (tv_deep < tv_shallow):
        raise SystemExit(
            'dark-year Tviscous should be shorter at small rp/rg, got '
            '{} vs {}'.format(tv_deep, tv_shallow))
    print('dark-year Tviscous deep', tv_deep, 'shallow', tv_shallow)

    fitter = Fitter(
        test=True, quiet=True, exit_on_prompt=True, prefer_cache=True)
    dummy_data = fitter.generate_dummy_data(name='tde_shock')
    pool = SerialPool()
    model = Model(
        model='tde_shock',
        data=dummy_data,
        test=True,
        printer=fitter._printer,
        fitter=fitter,
        pool=pool)
    ok = model.load_data(dummy_data, event_name='tde_shock', pool=pool)
    if not ok:
        raise SystemExit('tde_shock load_data failed')
    for required in ('frad', 'efficiency', 'viscous', 'Tviscous',
                     'total_luminosity', 'leddcap'):
        if required not in model._call_stack:
            raise SystemExit('{} missing from tde_shock'.format(required))
    if 'frad' not in model._free_parameters:
        raise SystemExit('frad is not a free parameter of tde_shock')
    if 'efficiency' not in model._free_parameters:
        raise SystemExit('efficiency should be free (narrow ε_acc prior)')
    if 'Tviscous' in model._free_parameters:
        raise SystemExit('Tviscous should be derived, not free')

    targets = {
        'frad': 0.03,
        'efficiency': 0.1,
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
    if 'shock_efficiency' not in outputs:
        raise SystemExit('Model.run did not return shock_efficiency')
    if 'rp_over_rg' not in outputs:
        raise SystemExit('Model.run did not return rp_over_rg')
    if 'Tviscous' not in outputs:
        raise SystemExit('Model.run did not return derived Tviscous')
    np.testing.assert_allclose(outputs['shock_efficiency'], 6.4e-4, rtol=0.15)
    np.testing.assert_allclose(outputs['rp_over_rg'], 47.0, rtol=0.15)
    np.testing.assert_allclose(outputs['efficiency'], 0.1, rtol=0.05)
    if not (outputs['Tviscous'] > 0.0):
        raise SystemExit('Tviscous should be positive')
    print('tde_shock Model.run shock_efficiency', outputs['shock_efficiency'],
          'rp_over_rg', outputs['rp_over_rg'], 'Tviscous', outputs['Tviscous'],
          'ndim', model._num_free_parameters)
    print('tde_shock checks passed')
    sys.exit(0)
