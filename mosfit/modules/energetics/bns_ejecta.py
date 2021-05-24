"""Definitions for the `BNSEjecta` class."""

import numpy as np
from astrocats.catalog.source import SOURCE
# import astropy.constants as c

from mosfit.constants import FOE, KM_CGS, M_SUN_CGS, C_CGS, G_CGS
from mosfit.modules.energetics.energetic import Energetic


# G_CGS = c.G.cgs.value


class BNSEjecta(Energetic):
    """
    Generate `mejecta`, `vejecta` and `kappa` from neutron star binary
    parameters.

    Includes tidal and shocked dynamical and disk wind ejecta following
    Dietrich+ 2017 and Coughlin+ 2019, with opacities from Sekiguchi+ 2016,
    Tanaka+ 2019, Metzger and Fernandez 2014, Lippuner+ 2017

    Also includes an ignorance parameter `alpha` for NS-driven winds to
    increase the fraction of blue ejecta: Mdyn_blue /= alpha
     - therefore NS surface winds turned off by setting alpha = 1
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2017CQGra..34j5014D'},
        {SOURCE.BIBCODE: '2019MNRAS.489L..91C'},
        {SOURCE.BIBCODE: '2013PhRvD..88b3007M'},
        {SOURCE.BIBCODE: '2016PhRvD..93l4046S'},
        {SOURCE.BIBCODE: '2014MNRAS.441.3444M'},
        {SOURCE.BIBCODE: '2017MNRAS.472..904L'},
        {SOURCE.BIBCODE: '2019LRR....23....1M'},
        {SOURCE.BIBCODE: '2020MNRAS.496.1369T'},
        {SOURCE.BIBCODE: '2018PhRvL.121i1102D'}
    ]

    def process(self, **kwargs):
        """Process module."""
        ckm = C_CGS / KM_CGS
        self._mchirp = kwargs[self.key('Mchirp')]
        self._q = kwargs[self.key('q')]
        # Mass of heavier NS
        self._m1 = self._mchirp * self._q**-0.6 * (self._q+1)**0.2
        # Mass of lighter NS
        self._m2 = self._m1*self._q
        self._m_total = self._m1 + self._m2
        # How much of disk is ejected
        self._disk_frac = kwargs[self.key('disk_frac')]
        # Max mass for non-rotating NS
        self._m_tov = kwargs[self.key('Mtov')]
        # Symmetric tidal polarizability
        self._LambdaSym = kwargs[self.key('LambdaSym')]
        # Fraction of blue ejecta from dynamical shocks (Coughlin+ 2019)
        # Here we are assuming remainder is a NS wind
        # So only applicable if merger product avoids prompt collapse
        self._alpha = kwargs[self.key('alpha')]
        # Opening angle
        self._cos_theta_open = kwargs[self.key('cos_theta_open')]
        self._kappa_red = kwargs[self.key('kappa_red')]
        self._kappa_blue = kwargs[self.key('kappa_blue')]

        theta_open = np.arccos(self._cos_theta_open)

        # Additional systematic error (useful when fitting)
        self._errMdyn = kwargs[self.key('errMdyn')]
        self._errMdisk = kwargs[self.key('errMdisk')]

        # # Symmetric mass ratio
        # eta = self._m1 * self._m2 / (self._m1 + self._m2)**2
        #
        # # Matrix elements for L1 and L2
        # A1 = 8/13*(1+7*eta-31*eta**2 + np.sqrt(1-4*eta)*(1+9*eta-11*eta**2))
        # A2 = 8/13*(1+7*eta-31*eta**2 - np.sqrt(1-4*eta)*(1+9*eta-11*eta**2))
        # # # Using approximation L \propto m^-6:
        # # A3 = 1
        # # A4 = -self._q**6
        # # Using definition of deltaLambdaTilde:
        # A3 = 1/2*(np.sqrt(1-4*eta)*(1-(13272*eta+8944*eta**2)/1319) +
        #         (1-(15910*eta+38850*eta**2+3380*eta**3)/1319))
        # A4 = 1/2*(np.sqrt(1-4*eta)*(1-(13272*eta+8944*eta**2)/1319) -
        #         (1-(15910*eta+38850*eta**2+3380*eta**3)/1319))
        #
        # A = np.array([[A1,A2],[A3,A4]])
        #
        # deltaLambda = 0
        # deltaLambda = self._dLambda
        # B = np.array([self._Lambda,deltaLambda])
        #
        # L1, L2 = np.linalg.solve(A,B)


        # Yagi and Yunes 2016 QUR to get L1 and L2 from Lsym
        a = 0.07550
        b = np.array([[-2.235, 0.8474],[10.45, -3.251],[-15.70, 13.61]])
        c = np.array([[-2.048, 0.5976],[7.941, 0.5658],[-7.360, -1.320]])
        n_ave = 0.743

        Fq = (1-(self._m2/self._m1)**(10./(3-n_ave)))/(1+(self._m2/self._m1)**(10./(3-n_ave)))

        nume = a
        denom = a

        Ls = self._LambdaSym

        for i in np.arange(3):
            for j in np.arange(2):
                nume += b[i,j]*(self._m2/self._m1)**(j+1)*Ls**(-(i+1)/5)

        for i in np.arange(3):
            for j in np.arange(2):
                denom += c[i,j]*(self._m2/self._m1)**(j+1)*Ls**(-(i+1)/5)

        La = Ls * Fq * nume / denom

        L1 = Ls - La
        L2 = Ls + La

        self._Lambda = 16./13 * ((self._m1 + 12*self._m2) * self._m1**4 * L1 +
                (self._m2 + 12*self._m1) * self._m2**4 * L2) / self._m_total**5

        # Approx radius of NS from De+ 2018
        self._radius_ns = 11.2 * self._mchirp * (self._Lambda/800)**(1./6.)

        # Compactness of each component (Maselli et al. 2013; Yagi & Yunes 2017)
        # C = (GM/Rc^2)
        C1 = 0.360 - 0.0355*np.log(L1) + 0.000705*np.log(L1)**2
        C2 = 0.360 - 0.0355*np.log(L2) + 0.000705*np.log(L2)**2

        self._R1 = (G_CGS * self._m1 * M_SUN_CGS / (C1 * C_CGS**2)) / 1e5
        self._R2 = (G_CGS * self._m2 * M_SUN_CGS / (C2 * C_CGS**2)) / 1e5


        # Baryonic masses, Gao 2019
        Mb1 = self._m1 + 0.08*self._m1**2
        Mb2 = self._m2 + 0.08*self._m2**2


        # Dynamical ejecta:

        # Fitting function from Dietrich and Ujevic 2017
        a_1 = -1.35695
        b_1 = 6.11252
        c_1 = -49.43355
        d_1 = 16.1144
        n = -2.5484
        Mejdyn = 1e-3* (a_1*((self._m2/self._m1)**(1/3)*(1-2*C1)/C1*Mb1 +
                        (self._m1/self._m2)**(1/3)*(1-2*C2)/C2*Mb2) +
                        b_1*((self._m2/self._m1)**n*Mb1 + (self._m1/self._m2)**n*Mb2) +
                        c_1*(Mb1-self._m1 + Mb2-self._m2) + d_1)
                        
        Mejdyn *= self._errMdyn
                        
        if Mejdyn < 0:
            Mejdyn = 0
            
            


        # Calculate fraction of ejecta with Ye<0.25 from fits to Sekiguchi 2016
        # Also consistent with Dietrich: mostly blue at M1/M2=1, all red by M1/M2=1.2.
        # And see Bauswein 2013, shocked (blue) component decreases with M1/M2
        a_4 = 14.8609
        b_4 = -28.6148
        c_4 = 13.9597

        f_red = min([a_4*(self._m1/self._m2)**2+b_4*(self._m1/self._m2)+c_4,1]) # fraction can't exceed 100%

        mejecta_red = Mejdyn * f_red
        mejecta_blue = Mejdyn * (1-f_red)


        # Velocity of dynamical ejecta
        a_2 = -0.219479
        b_2 = 0.444836
        c_2 = -2.67385

        vdynp = a_2*((self._m1/self._m2)*(1+c_2*C1) + (self._m2/self._m1)*(1+c_2*C2)) + b_2


        a_3 = -0.315585
        b_3 = 0.63808
        c_3 = -1.00757

        vdynz = a_3*((self._m1/self._m2)*(1+c_3*C1) + (self._m2/self._m1)*(1+c_3*C2)) + b_3

        vdyn = np.sqrt(vdynp**2+vdynz**2)

        # average velocity over angular ranges (< and > theta_open)

        theta1 = np.arange(0,theta_open,0.01)
        theta2 = np.arange(theta_open,np.pi/2,0.01)

        vtheta1 = np.sqrt((vdynz*np.cos(theta1))**2+(vdynp*np.sin(theta1))**2)
        vtheta2 = np.sqrt((vdynz*np.cos(theta2))**2+(vdynp*np.sin(theta2))**2)

        atheta1 = 2*np.pi*np.sin(theta1)
        atheta2 = 2*np.pi*np.sin(theta2)

        vejecta_blue = np.trapz(vtheta1*atheta1,x=theta1)/np.trapz(atheta1,x=theta1)
        vejecta_red = np.trapz(vtheta2*atheta2,x=theta2)/np.trapz(atheta2,x=theta2)

        vejecta_red *= ckm
        vejecta_blue *= ckm



        # Bauswein 2013, cut-off for prompt collapse to BH
        Mthr = (2.38-3.606*self._m_tov/self._radius_ns)*self._m_tov

        if self._m_total < Mthr:
            mejecta_blue /= self._alpha


        # Now compute disk ejecta following Coughlin+ 2019

        a_5 = -31.335
        b_5 = -0.9760
        c_5 = 1.0474
        d_5 = 0.05957

        logMdisk = np.max([-3, a_5*(1+b_5*np.tanh((c_5-self._m_total/Mthr)/d_5))])

        Mdisk = 10**logMdisk

        Mdisk *= self._errMdisk
        
        Mejdisk = Mdisk * self._disk_frac

        mejecta_purple = Mejdisk


        # Fit for disk velocity using Metzger and Fernandez
        vdisk_max = 0.15
        vdisk_min = 0.03
        vfit = np.polyfit([self._m_tov,Mthr],[vdisk_max,vdisk_min],deg=1)

        # Get average opacity of 'purple' (disk) component
        # Mass-averaged Ye as a function of remnant lifetime from Lippuner 2017
        # Lifetime related to Mtot using Metzger handbook table 3
        if self._m_total < self._m_tov:
            # stable NS
            Ye = 0.38
            vdisk = vdisk_max
        elif self._m_total < 1.2*self._m_tov:
            # long-lived (>>100 ms) NS remnant Ye = 0.34-0.38,
            # smooth interpolation
            Yfit = np.polyfit([self._m_tov,1.2*self._m_tov],[0.38,0.34],deg=1)
            Ye = Yfit[0]*self._m_total + Yfit[1]
            vdisk = vfit[0]*self._m_total + vfit[1]
        elif self._m_total < Mthr:
            # short-lived (hypermassive) NS, Ye = 0.25-0.34, smooth interpolation
            Yfit = np.polyfit([1.2*self._m_tov,Mthr],[0.34,0.25],deg=1)
            Ye = Yfit[0]*self._m_total + Yfit[1]
            vdisk = vfit[0]*self._m_total + vfit[1]
        else:
            # prompt collapse to BH, disk is red
            Ye = 0.25
            vdisk = vdisk_min

        # Convert Ye to opacity using Tanaka et al 2019 for Ye >= 0.25:
        a_6 = 2112.0
        b_6 = -2238.9
        c_6 = 742.35
        d_6 = -73.14

        kappa_purple = a_6*Ye**3 + b_6*Ye**2 + c_6*Ye + d_6

        vejecta_purple = vdisk * ckm

        vejecta_mean = (mejecta_purple*vejecta_purple + vejecta_red*mejecta_red +
                vejecta_blue*mejecta_blue) / (mejecta_purple + mejecta_red + mejecta_blue)

        kappa_mean = (mejecta_purple*kappa_purple + self._kappa_red*mejecta_red +
                self._kappa_blue*mejecta_blue) / (mejecta_purple + mejecta_red + mejecta_blue)

        return {self.key('mejecta_blue'): mejecta_blue,
                self.key('mejecta_red'): mejecta_red,
                self.key('mejecta_purple'): mejecta_purple,
                self.key('mejecta_dyn'): Mejdyn,
                self.key('mejecta_tot'): Mejdyn+mejecta_purple,
                self.key('vejecta_blue'): vejecta_blue,
                self.key('vejecta_red'): vejecta_red,
                self.key('vejecta_purple'): vejecta_purple,
                self.key('vejecta_mean'): vejecta_mean,
                self.key('kappa_purple'): kappa_purple,
                self.key('kappa_mean'): kappa_mean,
                self.key('M1'): self._m1,
                self.key('M2'): self._m2,
                self.key('R1'): self._R1,
                self.key('R2'): self._R2,
                self.key('radius_ns'): self._radius_ns,
                self.key('Lambda'): self._Lambda
                }
