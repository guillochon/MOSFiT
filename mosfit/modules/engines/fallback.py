"""Definitions for the `Fallback` class."""
import os

import astropy.constants as c
import numpy as np
from scipy.interpolate import interp1d

from mosfit.constants import C_CGS, DAY_CGS, FOUR_PI, M_SUN_CGS
from mosfit.modules.engines.engine import Engine

CLASS_NAME = 'Fallback'


class Fallback(Engine):
    """A tde engine."""

    def __init__(self, **kwargs):
        """Initialize module.

        Loads and interpolates tde simulation data. Simulation data is
        from Guillochon 2013 and can be found on astrocrash.net.
        The files in data directory have been converted from dm/dt space
        to dm/de space.
        """
        super(Fallback, self).__init__(**kwargs)

        G = c.G.cgs.value  # 6.67259e-8 cm3 g-1 s-2
        Mhbase = 1.0e6 * M_SUN_CGS  # this is the generic size of bh used

        self.EXTRAPOLATE = True

        # ------ DIRECTORY PARAMETERS -------

        # It is assumed that there are different files for each beta
        # (such as 2.500.dat for beta = 2.5)
        # The first row is energy, the second is dmde.

        self._gammas = ['4-3', '5-3']

        # dictionaries with gamma's as keys.
        self._beta_slope = {self._gammas[0]: [], self._gammas[1]: []}
        self._beta_yinter = {self._gammas[0]: [], self._gammas[1]: []}
        self._sim_beta = {self._gammas[0]: [], self._gammas[1]: []}
        self._mapped_time = {self._gammas[0]: [], self._gammas[1]: []}
        # for converting back from mapped time to actual times and doing
        # interpolation in actual time
        self._premaptime = {self._gammas[0]: [], self._gammas[1]: []}
        self._premapdmdt = {self._gammas[0]: [], self._gammas[1]: []}

        for g in self._gammas:

            dmdedir = (os.path.dirname(__file__)[:-15] + 'models/tde/data/' +
                       g + '/')

            # --------- GET SIMULATION BETAS -----------------
            sim_beta_files = os.listdir(dmdedir)

            self._sim_beta[g].extend([float(b[:-4]) for b in sim_beta_files])

            # ----- CREATE INTERPOLATION FUNCTIONS; FIND SLOPES & YINTERs -----
            time = {}
            dmdt = {}
            ipeak = {}
            mapped_time = {}
            # get dmdt and t for the lowest beta value
            # energy & dmde (cgs)
            e, d = np.loadtxt(dmdedir + sim_beta_files[0])
            # only convert dm/de --> dm/dt for mass that is bound to BH (e < 0)
            ebound = e[e < 0]
            dmdebound = d[e < 0]

            if min(dmdebound) < 0:  # shouldn't happen, just a check

                print('beta, gamma, negative dmde bound:', self._sim_beta[g],
                      g, dmdebound[dmdebound < 0])

            # calculate de/dt, time and dm/dt arrays
            # de/dt in log(/s), time in log(seconds), dm/dt in log(g/s)
            dedt = (1.0 / 3.0) * (-2.0 * ebound) ** (5.0 / 2.0) / \
                (2.0 * np.pi * G * Mhbase)
            time['lo'] = np.log10((2.0 * np.pi * G * Mhbase) *
                                  (-2.0 * ebound) ** (-3.0 / 2.0))
            dmdt['lo'] = np.log10(dmdebound * dedt)

            ipeak['lo'] = np.argmax(dmdt['lo'])

            # split time['lo'] & dmdt['lo'] into pre-peak and post-peak array
            time['lo'] = np.array([
                time['lo'][:ipeak['lo']],
                time['lo'][ipeak['lo']:]])  # peak in array 2
            dmdt['lo'] = np.array([
                dmdt['lo'][:ipeak['lo']],
                dmdt['lo'][ipeak['lo']:]])  # peak in array 2

            # will contain time/dmdt arrays
            # (split into pre & post peak times/dmdts)
            # for each beta value
            self._premaptime[g].append(np.copy(time['lo']))
            self._premapdmdt[g].append(np.copy(dmdt['lo']))

            for i in range(1, len(self._sim_beta[g])):
                # indexing this way bc calculating slope and yintercepts
                # BETWEEN each simulation beta

                e, d = np.loadtxt(dmdedir + sim_beta_files[i])
                # only convert dm/de --> dm/dt for mass bound to BH (e < 0)
                ebound = e[e < 0]
                dmdebound = d[e < 0]

                if min(dmdebound) < 0:  # shouldn't happen, just a check
                    print('beta, gamma, negative dmde bound:',
                          self._sim_beta[g], g, dmdebound[dmdebound < 0])

                # calculate de/dt, time and dm/dt arrays
                # de/dt in log(erg/s), time in log(seconds), dm/dt in log(g/s)
                dedt = (1.0 / 3.0) * (-2.0 * ebound) ** (5.0 / 2.0) / \
                    (2.0 * np.pi * G * Mhbase)
                time['hi'] = np.log10((2.0 * np.pi * G * Mhbase) *
                                      (-2.0 * ebound) ** (-3.0 / 2.0))
                dmdt['hi'] = np.log10(dmdebound * dedt)

                ipeak['hi'] = np.argmax(dmdt['hi'])

                # split time_hi and dmdt_hi into pre-peak and post-peak array
                # peak in 2nd array
                time['hi'] = np.array([time['hi'][:ipeak['hi']],
                                       time['hi'][ipeak['hi']:]])
                dmdt['hi'] = np.array([dmdt['hi'][:ipeak['hi']],
                                       dmdt['hi'][ipeak['hi']:]])
                # will contain time/dmdt arrays
                # (split into pre & post peak times/dmdts)
                # for each beta value
                self._premapdmdt[g].append(np.copy(dmdt['hi']))
                self._premaptime[g].append(np.copy(time['hi']))

                mapped_time['hi'] = []
                mapped_time['lo'] = []

                self._beta_slope[g].append([])
                self._beta_yinter[g].append([])
                self._mapped_time[g].append([])
                for j in [0, 1]:  # once before peak, once after peak
                    # choose more densely sampled curve to map times to 0-1
                    # less densely sampled curve will be interpolated to match
                    if len(time['lo'][j]) < len(time['hi'][j]):
                        # hi array more densely sampled
                        interp = 'lo'
                        nointerp = 'hi'
                    else:
                        # will also catch case where they have the same lengths
                        interp = 'hi'
                        nointerp = 'lo'
                    # map times from more densely sampled curves
                    # (both pre & post peak, might be from diff. dmdts)
                    # to 0 - 1
                    mapped_time[nointerp].append(
                        1. / (time[nointerp][j][-1] - time[nointerp][j][0]) *
                        (time[nointerp][j] - time[nointerp][j][0]))
                    mapped_time[interp].append(
                        1. / (time[interp][j][-1] - time[interp][j][0]) *
                        (time[interp][j] - time[interp][j][0]))

                    # ensure bounds are same for interp and nointerp
                    # before interpolation
                    # (should be 0 and 1 from above, but could be slightly off
                    # due to rounding errors in python)
                    mapped_time[interp][j][0] = 0
                    mapped_time[interp][j][-1] = 1
                    mapped_time[nointerp][j][0] = 0
                    mapped_time[nointerp][j][-1] = 1

                    func = interp1d(mapped_time[interp][j], dmdt[interp][j])
                    dmdtinterp = func(mapped_time[nointerp][j])

                    if interp == 'hi':
                        slope = ((dmdtinterp - dmdt['lo'][j]) /
                                 (self._sim_beta[g][i] - self._sim_beta[g][
                                     i - 1]))
                    else:
                        slope = ((dmdt['hi'][j] - dmdtinterp) /
                                 (self._sim_beta[g][i] - self._sim_beta[g][
                                     i - 1]))
                    self._beta_slope[g][-1].append(slope)

                    yinter1 = (dmdt[nointerp][j] - self._beta_slope[g][-1][j] *
                               self._sim_beta[g][i - 1])
                    yinter2 = (dmdtinterp - self._beta_slope[g][-1][j] *
                               self._sim_beta[g][i])
                    self._beta_yinter[g][-1].append((yinter1 + yinter2) / 2.0)
                    self._mapped_time[g][-1].append(
                        np.array(mapped_time[nointerp][j]))

                time['lo'] = np.copy(time['hi'])
                dmdt['lo'] = np.copy(dmdt['hi'])

    def process(self, **kwargs):
        """Process module."""
        beta_interp = True
        beta_outside_range = False

        Mhbase = 1.0e6  # in units of Msolar, this is generic Mh used
        # in astrocrash sims
        Mstarbase = 1.0  # in units of Msolar
        Rstarbase = 1.0  # in units of Rsolar

        # this is not beta, but rather a way to map beta_4-3 --> beta_5-3
        # b = 0 --> min disruption, b = 1 --> full disruption,
        # b = 2 --> max beta of sims
        self._b = kwargs['b']

        if 0 <= self._b < 1:
            # 0.6 is min disruption beta for gamma = 4/3
            # 1.85 is full disruption beta for gamma = 4/3
            beta43 = 0.6 + 1.25 * self._b  # 0.6 + (1.85 - 0.6)*b
            # 0.5 is min disruption beta for gamma = 5/3
            # 0.9 is full disruption beta for gamma = 5/3
            beta53 = 0.5 + 0.4 * self._b  # 0.5 + (0.9 - 0.5)*b

            self._betas = {'4-3': beta43, '5-3': beta53}

        elif 1 <= self._b <= 2:
            beta43 = 1.85 + 2.15 * (self._b - 1)
            beta53 = 0.9 + 1.6 * (self._b - 1)
            self._betas = {'4-3': beta43, '5-3': beta53}

        else:
            print('b outside range, bmin = 0; bmax = 2; b =', self._b)
            beta_outside_range = True

        # GET GAMMA VALUE

        gamma_interp = False

        self._Mstar = kwargs.get(self.key('starmass'), None)
        if self._Mstar <= 0.3 or self._Mstar >= 22:
            gammas = [self._gammas[1]]  # gamma = ['5-3']
            self._beta = self._betas['5-3']
        elif 1 <= self._Mstar <= 15:
            gammas = [self._gammas[0]]  # gamma = ['4-3']
            self._beta = self._betas['4-3']
        elif 0.3 < self._Mstar < 1:
            # region going from gamma = 5/3 to gamma = 4/3 as mass increases
            gamma_interp = True
            gammas = self._gammas
            # gfrac should == 0 for 4/3; == 1 for 5/3
            gfrac = (self._Mstar - 1.) / (0.3 - 1.)
            # beta_43 is always larger than beta_53
            self._beta = self._betas['5-3'] + (
                self._betas['4-3'] - self._betas['5-3']) * (1. - gfrac)
        elif 15 < self._Mstar < 22:
            # region going from gamma = 4/3 to gamma = 5/3 as mass increases
            gamma_interp = True
            gammas = self._gammas
            # gfrac should == 0 for 4/3; == 1 for 5/3
            gfrac = (self._Mstar - 15.) / (22. - 15.)

            # beta_43 is always larger than beta_53
            self._beta = self._betas['5-3'] + (
                self._betas['4-3'] - self._betas['5-3']) * (1. - gfrac)

        # try decoupling gamma from starmass
        '''
        self._scaled_gamma = kwargs['scaledgamma']
        # print (self._scaled_gamma)
        if self._scaled_gamma == 0.0: gammas = [self._gammas[0]]
        elif self._scaled_gamma == 1.0: gammas = [self._gammas[1]]
        else:
            gamma_interp = True
            gammas = self._gammas
            gfrac = self._scaled_gamma
        '''
        timedict = {}  # will hold time arrays for each g in gammas
        dmdtdict = {}  # will hold dmdt arrays for each g in gammas

        for g in gammas:
            # find simulation betas to interpolate between
            for i in range(len(self._sim_beta[g])):
                if self._betas[g] == self._sim_beta[g][i]:
                    # no need to interp, already have dmdt & t for this beta
                    beta_interp = False
                    interp_index_low = i
                    break

                if self._betas[g] < self._sim_beta[g][i]:
                    interp_index_high = i
                    interp_index_low = i - 1
                    beta_interp = True
                    break

            if not beta_outside_range and beta_interp:
                # ----------- LINEAR BETA INTERPOLATION --------------

                # get new dmdts  (2 arrays, pre & post peak (peak in array 2))
                # use interp_index_low bc of how slope and yintercept are saved
                # (slope[0] corresponds to between beta[0] and beta[1] etc.)
                dmdt = np.array([
                    self._beta_yinter[g][interp_index_low][0] +
                    self._beta_slope[g][interp_index_low][0] * self._betas[g],
                    self._beta_yinter[g][interp_index_low][1] +
                    self._beta_slope[g][interp_index_low][1] * self._betas[g]])

                # map mapped_times back to actual times, requires interpolation
                # in time
                # first for pre peak times

                time = []
                for i in [0, 1]:
                    # interp_index_low indexes beta
                    # mapped time between beta low and beta high
                    time_betalo = (
                        self._mapped_time[g][interp_index_low][i] *
                        (self._premaptime[g][interp_index_low][i][-1] -
                         self._premaptime[g][interp_index_low][i][0]) +
                        self._premaptime[g][interp_index_low][i][0])
                    time_betahi = (
                        self._mapped_time[g][interp_index_low][i] *
                        (self._premaptime[g][interp_index_high][i][-1] -
                         self._premaptime[g][interp_index_high][i][0]) +
                        self._premaptime[g][interp_index_high][i][0])

                    time.append(
                        time_betalo + (time_betahi - time_betalo) *
                        (self._betas[g] -
                         self._sim_beta[g][interp_index_low]) /
                        (self._sim_beta[g][interp_index_high] -
                         self._sim_beta[g][interp_index_low]))

                time = np.array(time)

                timedict[g] = time
                dmdtdict[g] = dmdt

            elif not beta_outside_range and not beta_interp:
                timedict[g] = np.copy(self._premaptime[g][interp_index_low])
                dmdtdict[g] = np.copy(self._premapdmdt[g][interp_index_low])

        # ---------------- GAMMA INTERPOLATION -------------------

        if gamma_interp:

            mapped_time = {'4-3': [], '5-3': []}

            time = []
            dmdt = []
            for j in [0, 1]:  # once before peak, once after peak
                # choose more densely sampled curve to map times to 0-1
                # less densely sampled curve will be interpolated to match
                if len(timedict['4-3'][j]) < len(timedict['5-3'][j]):
                    # gamma = 5/3 array more densely sampled
                    interp = '4-3'
                    nointerp = '5-3'
                else:
                    # will also catch case where they have the same lengths
                    interp = '5-3'
                    nointerp = '4-3'

                # map times from more densely sampled curves
                # (both pre & post peak, might be from diff. dmdts)
                # to 0 - 1
                mapped_time[nointerp].append(
                    1. / (timedict[nointerp][j][-1] -
                          timedict[nointerp][j][0]) *
                    (timedict[nointerp][j] - timedict[nointerp][j][0]))
                mapped_time[interp].append(
                    1. / (timedict[interp][j][-1] - timedict[interp][j][0]) *
                    (timedict[interp][j] - timedict[interp][j][0]))
                # ensure bounds same for interp & nointerp before interpolation
                # (they should be 0 and 1 from above, but could be slightly off
                # due to rounding errors in python)
                mapped_time[interp][j][0] = 0
                mapped_time[interp][j][-1] = 1
                mapped_time[nointerp][j][0] = 0
                mapped_time[nointerp][j][-1] = 1

                func = interp1d(mapped_time[interp][j], dmdtdict[interp][j])
                dmdtdict[interp][j] = func(mapped_time[nointerp][j])

                # recall gfrac = 0 --> gamma = 4/3, gfrac = 1 --> gamma 5/3
                if interp == '5-3':
                    # then mapped_time = mapped_time[nointerp] =
                    # mapped_time['4-3']
                    time53 = (mapped_time['4-3'][j] * (timedict['5-3'][j][-1] -
                                                       timedict['5-3'][j][0]) +
                              timedict['5-3'][j][0])
                    # convert back from logspace before adding to time array
                    time.extend(10 ** (timedict['4-3'][j] +
                                       (time53 - timedict['4-3'][j]) * gfrac))
                else:
                    # interp == '4-3'
                    time43 = (mapped_time['5-3'][j] * (timedict['4-3'][j][-1] -
                                                       timedict['4-3'][j][0]) +
                              timedict['4-3'][j][0])
                    # convert back from logspace before adding to time array
                    time.extend(10 ** (time43 +
                                       (timedict['5-3'][j] - time43) * gfrac))

                # recall gfrac = 0 --> gamma = 4/3, gfrac = 1 --> gamma 5/3
                # convert back from logspace before adding to dmdt array
                dmdt.extend(10 ** (dmdtdict['4-3'][j] +
                                   (dmdtdict['5-3'][j] -
                                    dmdtdict['4-3'][j]) * gfrac))

        else:  # gamma_interp == False
            # in this case, g will still be g from loop over gammas,
            # but there was only one gamma (no interpolation),
            # so g is the correct gamma
            # note that timedict[g] is a list not an array
            # no longer need a prepeak and postpeak array
            time = np.concatenate((timedict[g][0], timedict[g][1]))
            time = 10 ** time
            dmdt = np.concatenate((dmdtdict[g][0], dmdtdict[g][1]))
            dmdt = 10 ** dmdt

        time = np.array(time)
        dmdt = np.array(dmdt)

        # ----------- SCALE dm/dt TO BH & STAR MASS & STAR RADIUS -------------

        if 'dense_times' in kwargs:
            self._times = kwargs['dense_times']  # time in days
        else:
            print('in fallback, dense_times NOT in kwargs')
            self._times = kwargs['rest_times']

        # bh mass for dmdt's in astrocrash is 1e6 solar masses
        # dmdt ~ Mh^(-1/2)
        self._Mh = kwargs['bhmass']  # in units of solar masses

        # Assume that BDs below 0.1 solar masses are n=1 polytropes
        if self._Mstar < 0.1:
            Mstar_Tout = 0.1
        else:
            Mstar_Tout = self._Mstar

        # calculate Rstar from Mstar (using Tout et. al. 1996),
        # in Tout paper -> Z = 0.02 (now not quite solar Z) and ZAMS
        Z = 0.0134  # assume solar metallicity
        log10_Z_02 = np.log10(Z / 0.02)

        # Tout coefficients for calculating Rstar
        Tout_theta = (1.71535900 + 0.62246212 * log10_Z_02 - 0.92557761 *
                      log10_Z_02 ** 2 - 1.16996966 * log10_Z_02 ** 3 -
                      0.30631491 *
                      log10_Z_02 ** 4)
        Tout_l = (6.59778800 - 0.42450044 * log10_Z_02 - 12.13339427 *
                  log10_Z_02 ** 2 - 10.73509484 * log10_Z_02 ** 3 -
                  2.51487077 * log10_Z_02 ** 4)
        Tout_kpa = (10.08855000 - 7.11727086 * log10_Z_02 - 31.67119479 *
                    log10_Z_02 ** 2 - 24.24848322 * log10_Z_02 ** 3 -
                    5.33608972 * log10_Z_02 ** 4)
        Tout_lbda = (1.01249500 + 0.32699690 * log10_Z_02 - 0.00923418 *
                     log10_Z_02 ** 2 - 0.03876858 * log10_Z_02 ** 3 -
                     0.00412750 * log10_Z_02 ** 4)
        Tout_mu = (0.07490166 + 0.02410413 * log10_Z_02 + 0.07233664 *
                   log10_Z_02 ** 2 + 0.03040467 * log10_Z_02 ** 3 +
                   0.00197741 * log10_Z_02 ** 4)
        Tout_nu = 0.01077422
        Tout_eps = (3.08223400 + 0.94472050 * log10_Z_02 - 2.15200882 *
                    log10_Z_02 ** 2 - 2.49219496 * log10_Z_02 ** 3 -
                    0.63848738 * log10_Z_02 ** 4)
        Tout_o = (17.84778000 - 7.45345690 * log10_Z_02 - 48.9606685 *
                  log10_Z_02 ** 2 - 40.05386135 * log10_Z_02 ** 3 -
                  9.09331816 * log10_Z_02 ** 4)
        Tout_pi = (0.00022582 - 0.00186899 * log10_Z_02 + 0.00388783 *
                   log10_Z_02 ** 2 + 0.00142402 * log10_Z_02 ** 3 -
                   0.00007671 * log10_Z_02 ** 4)
        # caculate Rstar in units of Rsolar
        Rstar = ((Tout_theta * Mstar_Tout ** 2.5 + Tout_l *
                  Mstar_Tout ** 6.5 +
                  Tout_kpa * Mstar_Tout ** 11 + Tout_lbda *
                  Mstar_Tout ** 19 +
                  Tout_mu * Mstar_Tout ** 19.5) /
                 (Tout_nu + Tout_eps * Mstar_Tout ** 2 + Tout_o *
                  Mstar_Tout ** 8.5 + Mstar_Tout ** 18.5 + Tout_pi *
                  Mstar_Tout ** 19.5))

        dmdt = (dmdt * np.sqrt(Mhbase / self._Mh) *
                (self._Mstar / Mstarbase) ** 2.0 * (Rstarbase / Rstar) ** 1.5)
        # tpeak ~ Mh^(1/2) * Mstar^(-1)
        time = (time * np.sqrt(self._Mh / Mhbase) * (Mstarbase / self._Mstar) *
                (Rstar / Rstarbase) ** 1.5)

        time = time / DAY_CGS  # time is now in days to match self._times
        tfallback = np.copy(time[0])
        self._rest_t_explosion = kwargs['resttexplosion']  # units = days

        # ----------- EXTRAPOLATE dm/dt TO EARLY TIMES -------------
        # use power law to fit : dmdt = b*t^xi

        if self.EXTRAPOLATE and self._rest_t_explosion > self._times[0]:
            dfloor = min(dmdt)  # will be at late times if using James's
            # simulaiton data (which already has been late time extrap.)

            # not within 1% of floor, extrapolate --> NECESSARY?
            if dmdt[0] >= dfloor * 1.01:

                # try shifting time before extrapolation to make power law drop
                # off more suddenly around tfallback
                time = time + 0.9 * tfallback
                # this will ensure extrapolation will extend back to first
                # transient time.
                # requires self._rest_t_explosion > self._times[0]
                # time = (time - tfallback + self._rest_t_explosion -
                #        self._times[0])

                ipeak = np.argmax(dmdt)  # index of peak

                # the following makes sure there is enough prepeak sampling for
                # good extrapolation
                if ipeak < 1000:
                    prepeakfunc = interp1d(time[:ipeak], dmdt[:ipeak])
                    prepeaktimes = np.logspace(np.log10(time[0]),
                                               np.log10(time[ipeak - 1]), 1000)
                    # prepeaktimes = np.linspace(time[0], time[ipeak - 1],
                    #                           num=1000)
                    if prepeaktimes[-1] > time[ipeak - 1]:
                        prepeaktimes[-1] = time[ipeak - 1]
                    if prepeaktimes[0] < time[0]:
                        prepeaktimes[0] = time[0]
                    prepeakdmdt = prepeakfunc(prepeaktimes)
                else:
                    prepeaktimes = time[:ipeak]
                    prepeakdmdt = dmdt[:ipeak]

                start = 0

                # last index of first part of data used to get power law fit
                index1 = int(len(prepeakdmdt) * 0.1)
                # last index of second part of data used to get power law fit
                index2 = int(len(prepeakdmdt) * 0.15)

                t1 = prepeaktimes[start:index1]
                d1 = prepeakdmdt[start:index1]

                t2 = prepeaktimes[index2 - (index1 - start):index2]
                d2 = prepeakdmdt[index2 - (index1 - start):index2]

                # exponent for power law fit
                xi = np.log(d1 / d2) / np.log(t1 / t2)
                xiavg = np.mean(xi)

                # multiplicative factor for power law fit
                b1 = d1 / (t1 ** xiavg)

                bavg = np.mean(b1)

                tfloor = 0.01 + 0.9 * tfallback  # want first time ~0 (0.01)

                indexext = len(time[time < prepeaktimes[index1]])

                textp = np.linspace(tfloor, time[int(indexext)], num=ipeak * 5)
                dextp = bavg * (textp ** xiavg)

                time = np.concatenate((textp, time[int(indexext) + 1:]))

                time = time - 0.9 * tfallback  # shift back to original times

                dmdt = np.concatenate((dextp, dmdt[int(indexext) + 1:]))

        # try aligning first fallback time of simulation
        # (whatever first time is before early t extrapolation)
        # with parameter texplosion

        time = time - tfallback + self._rest_t_explosion

        tpeak = time[np.argmax(dmdt)]

        timeinterpfunc = interp1d(time, dmdt)

        lengthpretimes = len(np.where(self._times < time[0])[0])
        lengthposttimes = len(np.where(self._times > time[-1])[0])

        # this removes all extrapolation by interp1d by setting dmdtnew = 0
        # outside bounds of self._times
        dmdt1 = np.zeros(lengthpretimes)
        dmdt3 = np.zeros(lengthposttimes)
        # include len(self._times) instead of just using -lengthposttimes
        # for indexing in case lengthposttimes == 0
        dmdt2 = timeinterpfunc(self._times[lengthpretimes:(len(self._times) -
                                                           lengthposttimes)])
        dmdtnew = np.append(dmdt1, dmdt2)
        dmdtnew = np.append(dmdtnew, dmdt3)

        dmdtnew[dmdtnew < 0] = 0  # set floor for dmdt

        self._efficiency = kwargs['efficiency']
        # luminosities in erg/s
        luminosities = (self._efficiency * dmdtnew *
                        c.c.cgs.value * c.c.cgs.value)
        # -------------- EDDINGTON LUMINOSITY CUT -------------------
        # Assume solar metallicity for now

        # 0.2*(1 + X) = mean Thomson opacity
        kappa_t = 0.2 * (1 + 0.74)
        Ledd = (FOUR_PI * c.G.cgs.value * self._Mh * M_SUN_CGS *
                C_CGS / kappa_t)

        # 2 options for soft Ledd cuts, try both & see what fits stuff better
        # luminosities = np.where(
        #    luminosities > Ledd, (1. + np.log10(luminosities/Ledd)) * Ledd,
        #    luminosities)
        luminosities = (luminosities * Ledd / (luminosities + Ledd))

        return {'dense_luminosities': luminosities, 'Rstar': Rstar,
                'tpeak': tpeak, 'beta': self._beta, 'starmass': self._Mstar,
                'dmdt': dmdtnew, 'Ledd': Ledd}
