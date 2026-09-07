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

    # Two-component: cap accretion and shock separately, then sum.
    two = fb.process(frad=0.03, efficiency=0.03, bhmass=1.0e6, **shared)
    shock = np.asarray(two['dense_shock_luminosities'])
    acc = np.asarray(two['dense_luminosities'])
    np.testing.assert_allclose(
        two['shock_efficiency'] / 0.03, shock.max() / acc.max(), rtol=1e-10)
    cap = two['Ledd'] * 1.0
    acc_capped = acc * cap / (acc + cap)
    acc_capped = np.where(np.isnan(acc_capped), 0.0, acc_capped)
    # Super-Eddington shock so the collision cap is actually exercised.
    shock_in = shock * (10.0 * cap / max(float(np.max(shock)), 1.0))
    shock_capped = shock_in * cap / (shock_in + cap)
    shock_capped = np.where(np.isnan(shock_capped), 0.0, shock_capped)
    lc = LeddCap(name='leddcap', model=dummy)
    lc.set_attributes({
        'replacements': {'luminosities': 'acc_luminosities'},
        'wants_dense': True})
    lc._provide_dense = True
    lc_out = lc.process(
        dense_acc_luminosities=acc, Leddlim=1.0, Ledd=two['Ledd'],
        dense_indices=np.array([0, 1]))
    np.testing.assert_allclose(
        lc_out['dense_acc_luminosities'], acc_capped, rtol=1e-12)
    lc_shock = LeddCap(name='leddcap_shock', model=dummy)
    lc_shock.set_attributes({
        'replacements': {'luminosities': 'shock_luminosities'},
        'wants_dense': True})
    lc_shock._provide_dense = True
    shock_out = lc_shock.process(
        dense_shock_luminosities=shock_in, Leddlim=1.0, Ledd=two['Ledd'],
        dense_indices=np.array([0, 1]))
    np.testing.assert_allclose(
        shock_out['dense_shock_luminosities'], shock_capped, rtol=1e-12)
    if np.max(shock_out['dense_shock_luminosities']) > cap:
        raise SystemExit('capped shock exceeds Leddlim × Ledd')
    summed = shock_capped + acc_capped
    if np.max(summed) > 2.0 * cap:
        raise SystemExit('sum of separately capped terms exceeds 2 L_Edd')
    print('accretion and shock capped separately before sum')

    dy = DarkYear(name='Tviscous', model=dummy)
    dy_kw = dict(
        tfallback=40.0, tpeak=52.0, resttexplosion=0.0,
        tviscnorm=1.0, tviscslope=2.1, tviscoffset=0.0)
    t_pk = 92.0
    tv_ref = dy.process(rp_over_rg=47.0, **dy_kw)['Tviscous']
    tv_10 = dy.process(rp_over_rg=10.0, **dy_kw)['Tviscous']
    tv_hi = dy.process(rp_over_rg=1000.0, **dy_kw)['Tviscous']
    tv_lo = dy.process(rp_over_rg=2.0, **dy_kw)['Tviscous']
    tv_off = dy.process(
        rp_over_rg=47.0, tfallback=40.0, tpeak=52.0, resttexplosion=0.0,
        tviscnorm=1.0, tviscslope=2.1, tviscoffset=0.5)['Tviscous']
    np.testing.assert_allclose(tv_ref, 10.0 * t_pk, rtol=1e-10)
    np.testing.assert_allclose(
        tv_10, 10.0 ** (1.0 + 2.1 * np.log10(10.0 / 47.0)) * t_pk, rtol=1e-10)
    np.testing.assert_allclose(tv_hi, 100.0 * t_pk, rtol=1e-10)
    np.testing.assert_allclose(tv_lo, 10.0 ** (-1.5) * t_pk, rtol=1e-10)
    np.testing.assert_allclose(tv_off, 10.0 ** 1.5 * t_pk, rtol=1e-10)
    print('dark-year map', tv_ref, tv_10, tv_hi, tv_lo, tv_off)

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
    if 'leddcap' not in model._call_stack:
        raise SystemExit('leddcap missing from tde_shock')
    if 'leddcap_shock' not in model._call_stack:
        raise SystemExit('leddcap_shock missing from tde_shock')
    visc_inputs = [
        x if not isinstance(x, list) else x[0]
        for x in model._call_stack['viscous'].get('inputs', [])]
    if 'leddcap' not in visc_inputs:
        raise SystemExit('viscous should take capped acc from leddcap')
    tot_inputs = [
        x if not isinstance(x, list) else x[0]
        for x in model._call_stack['total_luminosity'].get('inputs', [])]
    if 'leddcap_shock' not in tot_inputs or 'viscous' not in tot_inputs:
        raise SystemExit(
            'total_luminosity should sum capped shock with viscous acc')
    photo_inputs = [
        x if not isinstance(x, list) else x[0]
        for x in model._call_stack['tde_photosphere'].get('inputs', [])]
    if 'total_luminosity' not in photo_inputs:
        raise SystemExit('tde_photosphere should read the separately capped sum')
    if 'frad' not in model._free_parameters:
        raise SystemExit('frad is not a free parameter of tde_shock')
    if 'efficiency' not in model._free_parameters:
        raise SystemExit('efficiency should be free (narrow ε_acc prior)')
    if 'Tviscous' in model._free_parameters:
        raise SystemExit('Tviscous should be derived, not free')
    if 'viscfac' in model._call_stack:
        raise SystemExit('viscfac should not be in tde_shock')
    if 'tviscoffset' not in model._call_stack:
        raise SystemExit('tviscoffset missing from tde_shock')
    if 'tviscoffset' in model._free_parameters:
        raise SystemExit('tviscoffset should be fixed')
    if not model._modules['tviscoffset']._fixed:
        raise SystemExit('tviscoffset module should be fixed')

    targets = {
        'frad': 0.03,
        'efficiency': 0.03,
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
    np.testing.assert_allclose(outputs['efficiency'], 0.03, rtol=0.05)
    run_cap = float(outputs['Leddlim']) * float(outputs['Ledd'])
    run_shock = np.asarray(outputs['dense_shock_luminosities'], dtype=float)
    run_total = np.asarray(outputs['dense_luminosities'], dtype=float)
    if np.max(run_shock) > run_cap:
        raise SystemExit('dense_shock_luminosities exceeds Leddlim × Ledd')
    if np.max(run_total) > 2.0 * run_cap:
        raise SystemExit('summed luminosity exceeds 2 Leddlim × Ledd')
    t_pk_run = float(outputs['tfallback']) + (
        float(outputs['tpeak']) - float(outputs['resttexplosion']))
    t_pk_run = max(t_pk_run, float(outputs['tfallback']))
    ratio = float(outputs['Tviscous']) / t_pk_run
    if not (5.0 <= ratio <= 15.0):
        raise SystemExit(
            'Tviscous/t_pk should be ~10 at 1e6 Msun beta=1, got {}'.format(
                ratio))
    model._modules['tviscoffset'].fix_value(0.5)
    outputs_off = model.run(x)
    np.testing.assert_allclose(
        outputs_off['Tviscous'], outputs['Tviscous'] * 10.0 ** 0.5,
        rtol=1e-8)
    print('tde_shock Model.run shock_efficiency', outputs['shock_efficiency'],
          'rp_over_rg', outputs['rp_over_rg'], 'Tviscous', outputs['Tviscous'],
          't_pk', t_pk_run, 'ratio', ratio,
          'ndim', model._num_free_parameters)
    print('tde_shock checks passed')
    sys.exit(0)
