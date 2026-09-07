"""Direct process() coverage for modules not always hit by model loads."""
from __future__ import print_function

import os
import sys
import tempfile
from collections import OrderedDict

import numpy as np

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    from mosfit.tests.dummies import DummyModel
    from mosfit.constants import BOL_BAND_INDEX
    from mosfit.modules.arrays.densetimes import DenseTimes
    from mosfit.modules.arrays.resttimes import RestTimes
    from mosfit.modules.constraints.constraint import Constraint
    from mosfit.modules.constraints.ia_constraints import IaConstraints
    from mosfit.modules.engines.blank import Blank
    from mosfit.modules.engines.exppow import ExpPow
    from mosfit.modules.engines.magnetar import Magnetar
    from mosfit.modules.engines.nickelcobalt import NickelCobalt
    from mosfit.modules.engines.rprocess import RProcess
    from mosfit.modules.engines.simplefallback import Simplefallback
    from mosfit.modules.energetics.homologous_expansion import HomologousExpansion
    from mosfit.modules.energetics.thin_shell import ThinShell
    from mosfit.modules.module import Module
    from mosfit.modules.objectives.likelihood import Likelihood
    from mosfit.modules.outputs.lightcurve import LightCurve
    from mosfit.modules.outputs.write import Write
    from mosfit.modules.parameters.constant import Constant
    from mosfit.modules.parameters.gaussian import Gaussian
    from mosfit.modules.parameters.kroupa import Kroupa
    from mosfit.modules.parameters.luminositydistance import LuminosityDistance
    from mosfit.modules.parameters.parameter import Parameter
    from mosfit.modules.parameters.powerlaw import PowerLaw
    from mosfit.modules.parameters.redshift import Redshift
    from mosfit.modules.photospheres.photosphere import Photosphere
    from mosfit.modules.photospheres.temperature_floor import TemperatureFloor
    from mosfit.modules.seds.blackbody_cutoff import BlackbodyCutoff
    from mosfit.modules.seds.sed import SED
    from mosfit.modules.transforms.transform import Transform
    from mosfit.modules.transforms.diffusion import Diffusion
    from mosfit.modules.utilities.operator import Operator
    from mosfit.modules.utilities.rename import Rename

    dummy = DummyModel()
    times = np.linspace(0.0, 40.0, 25)
    dense = np.linspace(0.0, 40.0, 80)

    rest = RestTimes(name='resttimes', model=dummy)
    out = rest.process(all_times=times, texplosion=-2.0, redshift=0.1)
    assert 'rest_times' in out

    dens = DenseTimes(name='densetimes', model=dummy)
    dout = dens.process(rest_times=out['rest_times'], resttexplosion=-1.8)
    assert 'dense_times' in dout

    mag = Magnetar(name='magnetar', model=dummy)
    mout = mag.process(
        dense_times=dense, Pspin=2.0, Bfield=1.0, Mns=1.4, thetaPB=0.5,
        resttexplosion=-1.0)
    assert mag.dense_key('luminosities') in mout

    ni = NickelCobalt(name='nickelcobalt', model=dummy)
    ni.process(
        dense_times=dense, fnickel=0.1, mejecta=2.0, resttexplosion=-1.0)

    ep = ExpPow(name='exppow', model=dummy)
    ep.process(
        dense_times=dense, alpha=1.0, beta=1.0, tpeak=10.0, lumscale=1e42,
        resttexplosion=-1.0)

    bl = Blank(name='blank', model=dummy)
    bl.process(dense_times=dense)

    sf = Simplefallback(name='simplefallback', model=dummy)
    sf.process(dense_times=dense, Lat1sec=1e43, ton=1.0, resttexplosion=-1.0)

    rp = RProcess(name='rprocess', model=dummy)
    rp.process(dense_times=dense, mejecta=0.05, resttexplosion=-1.0, vejecta=0.2)

    he = HomologousExpansion(name='homologous', model=dummy)
    assert 'vejecta' in he.process(kinetic_energy=1.0, mejecta=1.0)
    ts = ThinShell(name='thinshell', model=dummy)
    assert 'vejecta' in ts.process(kinetic_energy=1.0, mejecta=1.0)

    ia = IaConstraints(name='iac', model=dummy)
    ia.process(mejecta=1.0, vejecta=1e4, fnickel=0.5)
    Constraint(name='c0', model=dummy).process()

    par = Parameter(
        name='p', model=dummy, min_value=1.0, max_value=10.0, log=False)
    assert par.value(0.5) == 5.5
    assert 0.0 <= par.fraction(5.5) <= 1.0
    assert par.lnprior_pdf(0.5) == 0.0
    assert par.prior_icdf(0.3) == 0.3
    par.fix_value(3.0)
    assert par.process(fraction=0.1)['p'] == 3.0
    par2 = Parameter(
        name='q', model=dummy, min_value=1.0, max_value=100.0, log=True)
    assert par2.value(0.5) > 0
    g = Gaussian(
        name='g', model=dummy, min_value=1.0, max_value=10.0, mu=5.0, sigma=1.0)
    g.lnprior_pdf(0.5)
    g.prior_icdf(0.5)
    pl = PowerLaw(
        name='pl', model=dummy, min_value=1.0, max_value=10.0, alpha=2.0)
    pl.lnprior_pdf(0.5)
    pl.prior_icdf(0.5)
    kr = Kroupa(
        name='kr', model=dummy, min_value=0.02, max_value=10.0, log=False)
    kr.lnprior_pdf(0.2)
    kr.prior_icdf(0.5)
    const = Constant(name='c', model=dummy, value=3.0)
    z = Redshift(name='redshift', model=dummy, value=None, min_value=None,
                 max_value=None)
    z.receive_requests(lumdist=100.0)
    zout = z.process()
    assert 'redshift' in zout
    z.send_request('redshift')
    ld = LuminosityDistance(
        name='lumdist', model=dummy, value=None, min_value=None, max_value=None)
    ld.receive_requests(redshift=0.1)
    assert 'lumdist' in ld.process()
    ld.send_request('lumdist')

    sed = SED(name='sed', model=dummy)
    sed.receive_requests(band_wave_ranges=[[4000.0, 5000.0, 7000.0],
                                           [5000.0, 8000.0]])
    assert len(sed._sample_wavelengths) == 2
    added = sed.add_to_existing_seds(
        [np.ones(3), np.ones(3)], seds=[np.ones(3), np.ones(3)])
    assert np.allclose(added[0], 2.0)
    sed.send_request('sample_wavelengths')
    sed.set_data(12)

    nwav = 9
    sample = np.array([np.linspace(3000.0, 8000.0, nwav)])
    bbc = BlackbodyCutoff(name='bbc', model=dummy)
    bbc._sample_wavelengths = sample
    n = 6
    bbc.process(
        luminosities=np.full(n, 1e42),
        all_bands=['V'] * n,
        all_band_indices=np.zeros(n, dtype=int),
        all_frequencies=np.zeros(n),
        radiusphot=np.full(n, 1e14),
        temperaturephot=np.full(n, 8000.0),
        rest_times=np.linspace(0, 10, n),
        redshift=0.1,
        cutoff_wavelength=3000.0)

    tf = TemperatureFloor(name='tf', model=dummy)
    tf.process(
        luminosities=np.full(n, 1e42),
        resttexplosion=-1.0,
        rest_times=np.linspace(0, 10, n),
        temperature=5000.0,
        vejecta=1e4,
        mejecta=2.0)
    Photosphere(name='ph', model=dummy).process()

    tr = Transform(name='tr', model=dummy)
    tr._provide_dense = False
    tr.process(
        rest_times=np.linspace(1, 10, 8),
        resttexplosion=0.0,
        dense_times=np.linspace(0, 10, 20),
        dense_luminosities=np.ones(20) * 1e42)
    diff = Diffusion(name='diff', model=dummy)
    diff._provide_dense = False
    diff.process(
        rest_times=np.linspace(1, 10, 8),
        resttexplosion=0.0,
        dense_times=np.linspace(0, 10, 40),
        dense_luminosities=np.ones(40) * 1e42,
        kappa=0.1,
        kappagamma=0.1,
        mejecta=1.0,
        vejecta=1e4)

    op = Operator(name='op', model=dummy)
    op.set_attributes({'operands': ['a', 'b'], 'operator': '+', 'result': 'c'})
    assert op.process(a=np.array([1.0, 2.0]), b=np.array([3.0, 4.0]))['c'][0] == 4.0
    rn = Rename(name='rn', model=dummy)
    rn.set_attributes({'replacements': OrderedDict([('old', 'new')])})
    renamed = rn.process(old_key=1)
    assert 'new_key' in renamed

    like = Likelihood(name='like', model=dummy)
    like.process(
        fractions=[0.5],
        model_observations=np.array([18.0, 18.2]),
        observations=np.array([18.1, 18.0]),
        variances=np.array([0.1, 0.1]),
        upperlimits=np.array([False, False]),
        observed=np.array([True, True]),
        score_modifier=0.0)

    lc = LightCurve(name='fitlc', model=dummy)
    nobs = 4
    lc_out = lc.process(
        magnitudes=np.zeros(nobs),
        e_magnitudes=np.zeros(nobs),
        model_observations=np.full(nobs, 18.0),
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
        observation_types=np.array(['magnitude'] * nobs),
        abandvs=0.1,
        extra=1)
    assert 'model_variances' in lc_out
    dummy._fitter._limiting_magnitude = 20.0
    lc2 = LightCurve(name='fitlc2', model=dummy)
    lc2.process(
        magnitudes=np.zeros(nobs),
        e_magnitudes=np.zeros(nobs),
        model_observations=np.full(nobs, 18.0),
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
        observation_types=np.array(['magnitude'] * nobs),
        abandvs=0.1)
    Write(name='write', model=dummy).process()

    mod = Module(name='m', model=dummy)
    assert 'm' in repr(mod)
    mod.reset_preprocessed([])
    mod.send_request('x')
    assert mod.name() == 'm'
    mod.set_event_name('evt')
    mod.set_attributes({'replacements': OrderedDict([('a', 'b')]),
                        'wants_dense': True})
    assert mod.key('a') == 'b'
    mod.dense_key('luminosities')
    try:
        mod.prepare_input('missing')
        raise SystemExit('prepare_input should raise')
    except RuntimeError:
        pass
    mod.prepare_input(
        'luminosities',
        dense_luminosities=np.arange(5),
        dense_indices=np.array([0, 2]))
    mod.reset_unset_recommended_keys()
    mod.get_unset_recommended_keys()
    mod.get_bibcode()

    print('module unit tests passed')
    sys.exit(0)
