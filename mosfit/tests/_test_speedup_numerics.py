"""Numeric checks for vectorized SED/photometry/diagonal (not a pytest suite)."""
from __future__ import print_function

import os
import sys

import numpy as np

# Keep this import-only check independent of the rest of the file.
os.environ.setdefault('MOSFIT_PHOTOMETRY_DEBUG', '')


def test_import_no_torch():
    import sys as _sys
    before = {k for k in _sys.modules if k == 'torch' or k.startswith('torch.')}
    import mosfit  # noqa: F401
    after = {k for k in _sys.modules if k == 'torch' or k.startswith('torch.')}
    leaked = after - before
    assert not leaked, leaked
    print('import mosfit does not import torch')


def test_blackbody_matches_serial():
    from mosfit.modules.seds.blackbody import Blackbody
    from mosfit.constants import BOL_BAND_INDEX

    class DummyPool(object):
        size = 0
        comm = None

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def text(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    bb = Blackbody(name='blackbody', model=DummyModel())
    rng = np.random.RandomState(0)
    n_wav = 17
    sample = np.array([
        np.linspace(3000.0, 8000.0, n_wav),
        np.linspace(4000.0, 9000.0, n_wav),
    ])
    bb._sample_wavelengths = sample
    n = 40
    bi = np.array([0, 1, 0, 1, BOL_BAND_INDEX] * 8, dtype=int)
    lum = rng.uniform(1e40, 1e43, size=n)
    lum[3] = 0.0
    radius = rng.uniform(1e14, 1e15, size=n)
    temp = rng.uniform(5000.0, 20000.0, size=n)
    kwargs = {
        'luminosities': lum,
        'all_bands': ['g'] * n,
        'all_band_indices': bi,
        'all_frequencies': np.zeros(n),
        'radiusphot': radius,
        'temperaturephot': temp,
        'rest_times': np.linspace(0, 10, n),
        'redshift': 0.17,
    }
    out = bb.process(**kwargs)
    seds = out[bb.key('seds')]
    assert getattr(seds, 'ndim', 0) == 2
    assert seds.dtype != object

    xc = bb.X_CONST
    fc = bb.FLUX_CONST
    zp1 = 1.17
    Azp1 = __import__('astropy').units.Angstrom.cgs.scale / zp1
    ref = []
    for li in range(n):
        bii = bi[li]
        if bii == BOL_BAND_INDEX:
            ref.append(np.zeros(n_wav))
            continue
        if lum[li] == 0.0:
            ref.append(np.zeros(n_wav))
            continue
        rest_wavs = sample[bii] * Azp1
        rp = radius[li]
        tp = temp[li]
        row = fc * rp ** 2 / rest_wavs ** 5 / np.expm1(xc / rest_wavs / tp)
        row[np.isnan(row)] = 0.0
        ref.append(row)
    for li in range(n):
        np.testing.assert_allclose(np.asarray(seds[li]), ref[li], rtol=1e-10,
                                   atol=1e-20)
    print('Blackbody unique-band Planck matches serial formula')


def test_photometry_trapz():
    from mosfit.modules.observables.photometry import Photometry
    from mosfit.constants import MAG_FAC

    class DummyPool(object):
        size = 0
        comm = None

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def text(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    phot = Photometry.__new__(Photometry)
    phot._model = DummyModel()
    phot._pool = DummyPool()
    phot._printer = DummyPrinter()
    phot._preprocessed = True
    phot._name = 'photometry'
    phot._replacements = {}
    n_wav = 17
    wavs = np.linspace(4000.0, 8000.0, n_wav)
    trans = np.exp(-((wavs - 6000.0) / 800.0) ** 2)
    phot._band_wavelengths = [wavs]
    phot._transmissions = [trans]
    phot._band_areas = [[]]
    phot._band_offsets = np.array([0.0])
    phot._filter_integrals = np.array([
        Photometry.FLUX_STD * np.trapezoid(trans / wavs ** 2, wavs)])
    n = 12
    seds = np.stack([np.linspace(1e38, 2e38, n_wav) * (i + 1) for i in range(n)])
    kwargs = {
        'luminosities': np.ones(n),
        'all_band_indices': np.zeros(n, dtype=int),
        'observation_types': np.array(['magnitude'] * n),
        'observed': np.ones(n, dtype=bool),
        'lumdist': 100.0,
        'all_frequencies': np.zeros(n),
        'redshift': 0.1,
        'sample_wavelengths': [wavs],
        'seds': seds,
    }
    out = phot.process(**kwargs)
    zp1 = 1.1
    trans_i = np.interp(wavs, wavs, trans)
    yvals = trans_i * seds / zp1
    eff = np.trapezoid(yvals, wavs, axis=-1) / phot._filter_integrals[0]
    from mosfit.constants import FOUR_PI, MPC_CGS
    ldist = np.log10(FOUR_PI * (100.0 * MPC_CGS) ** 2)
    mags = -0.0 - MAG_FAC * (np.log10(eff) - ldist)
    np.testing.assert_allclose(out['model_observations'], mags, rtol=1e-10)
    out2 = phot.process(**kwargs)
    np.testing.assert_allclose(out2['model_observations'], mags, rtol=1e-10)
    assert phot._trans_on_sample[0] is not None
    print('Photometry band-grouped trapz matches serial formula')


def test_photometry_filter_cache_interp():
    """Cached filter interp matches on-the-fly np.interp when grids differ."""
    from mosfit.modules.observables.photometry import Photometry
    from mosfit.constants import FOUR_PI, MAG_FAC, MPC_CGS

    class DummyPool(object):
        size = 0
        comm = None

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def text(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    phot = Photometry.__new__(Photometry)
    phot._model = DummyModel()
    phot._pool = DummyPool()
    phot._printer = DummyPrinter()
    phot._preprocessed = True
    phot._name = 'photometry'
    phot._replacements = {}
    native = np.linspace(3500.0, 9000.0, 41)
    trans = np.exp(-((native - 6000.0) / 900.0) ** 2)
    sample = np.linspace(4000.0, 8000.0, 17)
    phot._band_wavelengths = [native]
    phot._transmissions = [trans]
    phot._band_areas = [[]]
    phot._band_offsets = np.array([0.0])
    phot._filter_integrals = np.array([
        Photometry.FLUX_STD * np.trapezoid(trans / native ** 2, native)])
    n = 8
    seds = np.stack([np.linspace(1e38, 2e38, 17) * (i + 1) for i in range(n)])
    kwargs = {
        'luminosities': np.ones(n),
        'all_band_indices': np.zeros(n, dtype=int),
        'observation_types': np.array(['magnitude'] * n),
        'observed': np.ones(n, dtype=bool),
        'lumdist': 80.0,
        'all_frequencies': np.zeros(n),
        'redshift': 0.05,
        'sample_wavelengths': [sample],
        'seds': seds,
    }
    out = phot.process(**kwargs)
    trans_i = np.interp(sample, native, trans)
    zp1 = 1.05
    yvals = trans_i * seds / zp1
    eff = np.trapezoid(yvals, sample, axis=-1) / phot._filter_integrals[0]
    ldist = np.log10(FOUR_PI * (80.0 * MPC_CGS) ** 2)
    mags = - MAG_FAC * (np.log10(eff) - ldist)
    np.testing.assert_allclose(out['model_observations'], mags, rtol=1e-10)
    np.testing.assert_allclose(phot._trans_on_sample[0], trans_i, rtol=0,
                               atol=0)
    out2 = phot.process(**kwargs)
    np.testing.assert_allclose(out2['model_observations'], mags, rtol=1e-10)
    print('Photometry filter cache matches np.interp')


def test_diagonal_residuals():
    from mosfit.modules.arrays.diagonal import Diagonal

    class DummyPool(object):
        size = 0

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    d = Diagonal(name='diagonal', model=DummyModel())
    n = 8
    model = np.array([18.0, 19.5, 17.0, 20.0, 18.2, 16.0, 21.0, 18.5])
    mags = np.array([18.1, 19.0, 17.2, 21.0, 18.0, 16.5, 22.0, 18.4])
    ul = np.array([False, True, False, True, False, False, True, False])
    d._preprocessed = True
    d._observed = np.ones(n, dtype=bool)
    d._o_types = np.array(['magnitude'] * n)
    d._mags_f = mags.astype(float)
    d._lums_f = np.full(n, np.nan)
    d._cts_f = np.full(n, np.nan)
    d._fds_f = np.full(n, np.nan)
    d._upper_limits = ul
    d._e_l_mags_f = np.full(n, 0.1)
    d._e_u_mags_f = np.full(n, 0.2)
    d._e_l_lums_f = np.full(n, 0.1)
    d._e_u_lums_f = np.full(n, 0.2)
    d._e_l_cts_f = np.full(n, 0.1)
    d._e_u_cts_f = np.full(n, 0.2)
    d._e_l_fds_f = np.full(n, 0.1)
    d._e_u_fds_f = np.full(n, 0.2)
    out = d.process(model_observations=model)
    from math import isnan
    ref = []
    for x, y, u in zip(model, mags, ul):
        if (not u and y is not None) or (not isnan(x) and y is not None and x < y):
            ref.append(abs(x - y))
        else:
            ref.append(0.0)
    np.testing.assert_allclose(out['kresiduals'], np.array(ref), rtol=0, atol=0)
    print('Diagonal residuals match serial definition')


def test_mm83():
    from mosfit.modules.seds.losextinction import LOSExtinction

    class DummyPool(object):
        size = 0

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    los = LOSExtinction(name='ext', model=DummyModel())
    waves = np.array([2.0, 50.0, 200.0, 800.0, 2000.0])
    nh = 1.0e21
    vec = los.mm83(nh, waves)
    y = np.array([los.H_C_CGS / (x * los.ANG_CGS * los.KEV_CGS) for x in waves])
    i = np.array([np.searchsorted(los._mm83[:, 0], x) - 1 for x in y])
    al = [1.0e-24 * (los._mm83[x, 1] + los._mm83[x, 2] * y[j] +
                     los._mm83[x, 3] * y[j] ** 2) / y[j] ** 3
          for j, x in enumerate(i)]
    al = [al[j] if x < los._min_xray
          else los._almin * (los._min_xray / x) ** 3
          for j, x in enumerate(y)]
    al = [al[j] if x > los._max_xray
          else los._almax * (los._max_xray / x) ** 3
          for j, x in enumerate(y)]
    ref = nh * np.array(al)
    np.testing.assert_allclose(vec, ref, rtol=1e-12)
    print('mm83 vectorized matches list-comprehension formula')


def _fallback_dummy_model():
    class DummyPool(object):
        size = 0
        comm = None

        def is_master(self):
            return True

    class DummyPrinter(object):
        def message(self, *a, **k):
            return ''

        def text(self, *a, **k):
            return ''

        def prt(self, *a, **k):
            return ''

    class DummyModel(object):
        def pool(self):
            return DummyPool()

        def printer(self):
            return DummyPrinter()

    return DummyModel()


def test_fallback_golden():
    """TDE Fallback engine matches pre-NumPyization reference draws."""
    from mosfit.modules.engines.fallback import Fallback

    fb = Fallback(name='fallback', model=_fallback_dummy_model())
    fb._provide_dense = True
    times = np.unique(np.concatenate((
        [0.0], np.logspace(-6, 2, 100) + 10.0, np.linspace(0, 120, 40))))
    cases = [
        dict(b=0.7, starmass=0.5, bhmass=1e7, resttexplosion=10.0,
             efficiency=0.1, Leddlim=1.0, dense_times=times),
        dict(b=1.2, starmass=8.0, bhmass=1e6, resttexplosion=5.0,
             efficiency=0.01, Leddlim=2.0, dense_times=times),
        dict(b=0.4, starmass=0.5, bhmass=5e6, resttexplosion=12.0,
             efficiency=0.05, Leddlim=1.0, dense_times=times),
        dict(b=0.2, starmass=18.0, bhmass=2e6, resttexplosion=8.0,
             efficiency=0.2, Leddlim=1.5, dense_times=times),
    ]
    expected = [
        dict(Rstar=0.45793094821444064, tpeak=62.85075767517639,
             beta=0.9785714285714286, tfallback=43.53189249267144,
             Ledd=1.4366896299274834e+45, lum_sum=2.7710499828301767e+46,
             lum_max=7.638370418643626e+44, dmdt_sum=5.6062764714291976e+26,
             lum0=[1.0857703619278655e+38, 1.6528214942483027e+39,
                   2.2610565163202123e+40, 2.792625467582705e+41,
                   5.164827934172274e+41, 5.164828782023136e+41,
                   5.1648298032624925e+41, 5.164831033348916e+41],
             lum40=[5.1681866533337065e+41, 5.1688743709498475e+41,
                    5.1697027291657415e+41, 5.170700489442526e+41,
                    5.171902295044171e+41, 5.17334987387902e+41,
                    5.175093487322971e+41, 5.177193675326944e+41]),
        dict(Rstar=3.2673242592313327, tpeak=22.12046500273871,
             beta=2.28, tfallback=4.251391663150998,
             Ledd=1.4366896299274836e+44, lum_sum=3.642094706671135e+46,
             lum_max=2.826626435272511e+44, dmdt_sum=7.528376770927641e+28,
             lum0=[0.0, 7.507834342980549e+38, 4.983278396334756e+43,
                   2.5590165046518173e+44, 2.6472293354960365e+44,
                   2.6472293532181025e+44, 2.6472293745643895e+44,
                   2.6472294002760626e+44],
             lum40=[2.647299518815636e+44, 2.647313883896991e+44,
                    2.647331184264855e+44, 2.6473520191039516e+44,
                    2.647377109642813e+44, 2.6474073238910075e+44,
                    2.6474437063358467e+44, 2.647487513568258e+44]),
        dict(Rstar=0.45793094821444064, tpeak=47.78384362963816,
             beta=0.7857142857142858, tfallback=40.54390246908036,
             Ledd=7.183448149637417e+44, lum_sum=8.157357339457045e+45,
             lum_max=2.571276983096958e+44, dmdt_sum=2.5095187370787093e+26,
             lum0=[7.110072264678295e+35, 2.0908697837684794e+37,
                   5.294665830233182e+38, 1.1692555638944066e+40,
                   2.482753579129413e+40, 2.482754104413379e+40,
                   2.482754737119771e+40, 2.482755499216862e+40],
             lum40=[2.4848344659498873e+40, 2.4852605399406297e+40,
                    2.4857737475641795e+40, 2.4863919079539e+40,
                    2.4871364843176347e+40, 2.488033329163258e+40,
                    2.4891135819251427e+40, 2.490414750158064e+40]),
        dict(Rstar=5.300277743392906, tpeak=34.000670082779045,
             beta=0.7342857142857143, tfallback=30.681094958527872,
             Ledd=2.873379259854967e+44, lum_sum=2.740613590552974e+46,
             lum_max=4.19074542001477e+44, dmdt_sum=2.1255254353845016e+27,
             lum0=[3.7164217421970233e+40, 4.526128916747076e+41,
                   4.744151844041845e+42, 4.0623125502845477e+43,
                   6.53510243562584e+43, 6.535103210918421e+43,
                   6.535104144761026e+43, 6.535105269577687e+43],
             lum40=[6.538173461581153e+43, 6.538802208864473e+43,
                    6.539559508483632e+43, 6.540471636905956e+43,
                    6.541570238397375e+43, 6.542893420128521e+43,
                    6.544487070050231e+43, 6.546406442562971e+43]),
    ]
    for i, (kw, exp) in enumerate(zip(cases, expected)):
        out = fb.process(**kw)
        lum = np.asarray(out['dense_luminosities'], dtype=float)
        np.testing.assert_allclose(out['Rstar'], exp['Rstar'], rtol=1e-12)
        np.testing.assert_allclose(out['tpeak'], exp['tpeak'], rtol=1e-12)
        np.testing.assert_allclose(out['beta'], exp['beta'], rtol=1e-12)
        np.testing.assert_allclose(out['tfallback'], exp['tfallback'],
                                   rtol=1e-12)
        np.testing.assert_allclose(out['Ledd'], exp['Ledd'], rtol=1e-12)
        np.testing.assert_allclose(np.sum(lum), exp['lum_sum'], rtol=1e-10)
        np.testing.assert_allclose(np.max(lum), exp['lum_max'], rtol=1e-10)
        np.testing.assert_allclose(np.sum(out['dmdt']), exp['dmdt_sum'],
                                   rtol=1e-10)
        np.testing.assert_allclose(lum[:8], exp['lum0'], rtol=1e-10)
        np.testing.assert_allclose(lum[40:48], exp['lum40'], rtol=1e-10)
    print('Fallback engine matches golden draws')


def _viscous_quadrature(dense_t, dense_l, times, tvisc, tb, t_end):
    """Previous 1000-point trapezoid on log-spaced nodes (physics baseline)."""
    from mosfit.modules.transforms.viscous import Viscous

    times = np.asarray(times, dtype=float)
    uniq_times = np.unique(times[(times >= tb) & (times <= t_end)])
    lu = len(uniq_times)
    num = int(Viscous.N_INT_TIMES / 2.0)
    lsp = np.logspace(
        np.log10(tvisc / t_end) + Viscous.MIN_LOG_SPACING, 0, num)
    xm = np.unique(np.concatenate((lsp, 1 - lsp)))
    int_times = np.clip(
        tb + (uniq_times.reshape(lu, 1) - tb) * xm, tb, t_end)
    int_lums = np.interp(int_times, dense_t, dense_l)
    int_args = int_lums * np.exp(
        (int_times - int_times[:, -1].reshape(lu, 1)) / tvisc)
    int_args[np.isnan(int_args)] = 0.0
    uniq_lums = np.trapezoid(int_args, int_times) / tvisc
    return uniq_lums[np.searchsorted(uniq_times, times)]


def test_viscous_matches_interp1d():
    """Exponential recurrence matches analytic kernel and old quadrature."""
    from mosfit.modules.transforms.viscous import Viscous, viscous_exp_filter
    from mosfit.modules.transforms.transform import Transform

    rest_t = 10.0
    dense = np.unique(np.concatenate((
        [0.0], np.logspace(-6, 2, 80) + rest_t, np.linspace(0, 80, 40))))
    lums = 1e43 * np.exp(-np.clip(dense - rest_t, 0, None) / 15.0) * (
        dense >= rest_t)
    rest_times = np.linspace(0, 80, 120)
    tvisc = 3.5
    kwargs = dict(
        rest_times=rest_times,
        resttexplosion=rest_t,
        dense_times=dense,
        dense_luminosities=lums,
        Tviscous=tvisc,
    )

    v = Viscous(name='viscous', model=_fallback_dummy_model())
    v._provide_dense = True
    out = v.process(**kwargs)
    y = np.asarray(out['dense_luminosities'])
    out2 = v.process(**kwargs)
    np.testing.assert_allclose(y, out2['dense_luminosities'], rtol=0, atol=0)

    Transform.process(v, **kwargs)
    dense_t = np.asarray(v._dense_times_since_exp, dtype=float)
    dense_l = np.asarray(v._dense_luminosities, dtype=float)
    min_te = float(np.min(dense_t))
    tb = max(0.0, min_te)
    t_end = float(dense_t[-1])
    times = np.asarray(v._times_to_process, dtype=float)

    y_py = viscous_exp_filter.py_func(
        dense_t, dense_l, np.unique(times[(times >= tb) & (times <= t_end)]),
        tvisc, tb, t_end)
    y_jit = viscous_exp_filter(
        dense_t, dense_l, np.unique(times[(times >= tb) & (times <= t_end)]),
        tvisc, tb, t_end)
    np.testing.assert_allclose(y_jit, y_py, rtol=1e-12, atol=0)

    quad = _viscous_quadrature(dense_t, dense_l, times, tvisc, tb, t_end)
    scale = np.maximum(np.abs(quad), 1e-30)
    rel = np.abs(y - quad) / scale
    max_rel = float(np.max(rel))
    print('Viscous recurrence vs 1000-pt quadrature max rel', max_rel)
    np.testing.assert_allclose(y, quad, rtol=3e-4, atol=0)

    for tau in (1.0e-3, 0.075, 1.0, 100.0):
        kwargs['Tviscous'] = tau
        y_tau = np.asarray(v.process(**kwargs)['dense_luminosities'])
        q_tau = _viscous_quadrature(dense_t, dense_l, times, tau, tb, t_end)
        rel_tau = float(np.max(np.abs(y_tau - q_tau) /
                               np.maximum(np.abs(q_tau), 1e-30)))
        print('Viscous recurrence vs quadrature tau', tau, 'max rel', rel_tau)
        np.testing.assert_allclose(y_tau, q_tau, rtol=1e-3, atol=0)

    # Constant L: L_out(t) = L0 (1 - exp(-(t-tb)/τ)).
    L0 = 1.0e42
    t_grid = np.linspace(0.0, 40.0, 81)
    L_const = np.full_like(t_grid, L0)
    tau = 4.0
    y_const = viscous_exp_filter(t_grid, L_const, t_grid, tau, 0.0, t_grid[-1])
    analytic = L0 * (1.0 - np.exp(-t_grid / tau))
    np.testing.assert_allclose(y_const, analytic, rtol=1e-12, atol=1e-6 * L0)
    print('Viscous exponential recurrence matches serial formula')


def test_viscous_loader_name_is_importable():
    """MOSFiT's dynamic loader name must be a real package path.

    Concatenating ``transforms`` + ``viscous`` without a dot made Numba's
    disk cache pickle ``mosfit.modules.transformsviscous``, which later
    processes cannot import.
    """
    import importlib
    import importlib.machinery
    from mosfit.model import task_module_qualname

    name = task_module_qualname('transforms', 'viscous')
    assert name == 'mosfit.modules.transforms.viscous'
    mod = importlib.import_module(name)
    assert hasattr(mod, 'viscous_exp_filter')

    path = os.path.join(
        os.path.dirname(mod.__file__), 'viscous.py')
    loaded = importlib.machinery.SourceFileLoader(name, path).load_module()
    t = np.array([0.0, 1.0, 2.0])
    lums = np.array([0.0, 1.0, 0.0])
    y = loaded.viscous_exp_filter(t, lums, t, 1.0, 0.0, 2.0)
    y2 = mod.viscous_exp_filter(t, lums, t, 1.0, 0.0, 2.0)
    np.testing.assert_allclose(y, y2, rtol=0, atol=0)
    print('Viscous loader module name is importable')


def test_cwd_override_module_is_importable():
    """CWD module overrides must pickle under a path spawn workers can import."""
    import importlib
    import pickle
    import shutil
    import tempfile
    from mosfit.model import (
        _is_cwd_override, ensure_cwd_on_sys_path, task_module_qualname)

    kinds = 'transforms'
    leaf = 'cwdspawnprobe'
    pkg = os.path.join(
        os.path.dirname(importlib.import_module(
            'mosfit.modules.transforms.viscous').__file__),
        leaf + '.py')
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    old_path = list(sys.path)
    extra_keys = []
    try:
        os.chdir(tmp)
        os.makedirs(os.path.join('modules', kinds))
        path = os.path.join('modules', kinds, leaf + '.py')
        with open(path, 'w') as f:
            f.write('class Probe(object):\n    marker = 17\n')
        assert _is_cwd_override(path, pkg)
        ensure_cwd_on_sys_path()
        name = task_module_qualname(kinds, leaf, from_cwd=True)
        assert name == 'modules.transforms.cwdspawnprobe'
        extra_keys = [
            k for k in ('modules', 'modules.transforms', name)
            if k not in sys.modules]
        mod = importlib.import_module(name)
        blob = pickle.dumps(mod.Probe)
        for key in ('modules.transforms.cwdspawnprobe',
                    'modules.transforms', 'modules'):
            sys.modules.pop(key, None)
        cls = pickle.loads(blob)
        assert cls.marker == 17
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for key in extra_keys:
            sys.modules.pop(key, None)
        shutil.rmtree(tmp, ignore_errors=True)
    print('CWD override module name is importable')


if __name__ == '__main__':
    test_import_no_torch()
    test_blackbody_matches_serial()
    test_photometry_trapz()
    test_photometry_filter_cache_interp()
    test_diagonal_residuals()
    test_mm83()
    test_fallback_golden()
    test_viscous_matches_interp1d()
    test_viscous_loader_name_is_importable()
    test_cwd_override_module_is_importable()
    print('all numeric tests passed')
    sys.exit(0)
