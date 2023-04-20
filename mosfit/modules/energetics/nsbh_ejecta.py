"""Definitions for the `NSBHEjecta` class."""

import numpy as np
from astrocats.catalog.source import SOURCE

from mosfit.constants import FOE, KM_CGS, M_SUN_CGS, C_CGS, G_CGS
from mosfit.modules.energetics.energetic import Energetic


class NSBHEjecta(Energetic):
    """
    Generate `mejecta` and `vejecta` from black hole - neutron star
    binary parameters.

    Includes tidal dynamical and both thermal and magnetic disk wind ejecta
    following Kawaguchi et al. 2016, Foucart et al. 2018, Kruger & Foucart
    2020, and Fernandez et al. 2020. Opacities from Tanaka & Hotokezaka 2013,
    Kawaguchi et al. 2016, Kasen et al. 2017 and Tanaka et al. 2020.

    Includes an ignorance parameter fMag for magnetically-driven winds
    (Fernandez et al. 2019, Christie et al. 2019).
    """

    _REFERENCES = [
        {SOURCE.BIBCODE: '2018PhRvD..98h1501F'},
        {SOURCE.BIBCODE: '2020PhRvD.101j3002K'},
        {SOURCE.BIBCODE: '2016ApJ...825...52K'},
        {SOURCE.BIBCODE: '2018PhRvD..97b3009K'},
        {SOURCE.BIBCODE: '2020MNRAS.496.1369T'},
        {SOURCE.BIBCODE: '2020MNRAS.497.3221F'},
        {SOURCE.BIBCODE: '2019MNRAS.482.3373F'},
        {SOURCE.BIBCODE: '2020PhRvD.101h3029F'},
        {SOURCE.BIBCODE: '2008PhRvD..77b1502F'},
        {SOURCE.BIBCODE: '2017PhR...681....1Y'},
        {SOURCE.BIBCODE: '2019MNRAS.490.4811C'}
    ]

    def process(self, **kwargs):
        """Process module."""
        ckm = C_CGS / KM_CGS
        self._mchirp = kwargs[self.key('Mchirp')]
        self._q = kwargs[self.key('q')]
        self._LambdaEff = kwargs[self.key('LambdaEff')] # Effective tidal deformability

        # BH properties
        self._Mbh = self._mchirp * self._q**-0.6 * (self._q+1)**0.2 # Mass.
        self._chi = kwargs[self.key('chi')] # Orbit-aligned spin.

        # NS properties
        self._Mns = self._Mbh * self._q # Mass (via mass ratio q).
        self._Lambda_ns = 13. / 16. * self._LambdaEff * (self._Mbh + self._Mns) ** 5. / ((self._Mns + 12. * self._Mbh) * self._Mns ** 4.) # Tidal deformability
        Cns = 0.360 - 0.0355 * np.log(self._Lambda_ns) + 0.000705 * np.log(self._Lambda_ns) ** 2. # NS compactness, cf. Yagi & Yunes (2017).
        self._radius_ns = (G_CGS * self._Mns * M_SUN_CGS / (Cns * C_CGS ** 2.)) / 1E5 # NS radius (recorded but not used again)

        # Binary properties
        self._M_total = self._Mbh + self._Mns   # Total mass.
        eta = self._q ** (-1.0) / (1. + self._q ** (-1.0)) ** 2.0 # Parameterisation of q from Foucart et al. (2018).


        # Additional systematic error (useful when fitting)
        self._errMdyn = kwargs[self.key('errMdyn')]
        self._errMdisk = kwargs[self.key('errMdisk')]

        # Calculate the ISCO
        Z1 = 1.0 + (1.0 - self._chi ** 2.0) ** (1.0 / 3.0) * ((1.0 + self._chi) ** (1.0 / 3.0) + (1.0 - self._chi) ** (1.0 / 3.0))
        Z2 = np.sqrt(3.0 * self._chi ** 2.0 + Z1 ** 2.0)
        Risco = 3.0 + Z2 - np.sign(self._chi) * np.sqrt((3.0 - Z1) * (3. + Z1 + 2.0 * Z2)) # Normalised Risco, equal to Risco/Mbh

        # Remnant mass fitting parameters (Foucart et al. 2018).
        a = 0.406
        b = 0.139
        c = 0.255
        d = 1.761

        # Dynamical ejecta mass fitting parameters (Kruger & Foucart 2021).
        a1 = 0.007116
        a2 = 0.001436
        a4 = -0.02762
        n1 = 0.8636
        n2 = 1.6840

        # Baryon mass of the neutron star (Eqn 7 from Kruger & Foucart 2021)
        Mb = self._Mns * (1.0 + (0.6 * Cns) / (1.0 - 0.5 * Cns))

        # Remnant mass outside of the event horizon
        Mrem = (max(a * (1 - 2 * Cns) / eta ** (1.0 / 3.0) - b * Risco * Cns / eta + c, 0)) ** d * Mb

        # Dynamical ejecta mass
        Mdyn = max((a1 / self._q ** n1 * (1.0 - 2. * Cns) / Cns - a2 / self._q ** n2 * Risco + a4) * Mb, 0) *1.68
        
        # Average dynamical ejecta velocity from Kawaguchi et al. (2016)
        Vdyn = (0.01533 / self._q + 0.1907) * ckm

        # Disc mass
        Mdisc = Mrem - Mdyn
        
        ##########
        # Fiducial values of post-merger disc radii collated in Fernandez et al. (2020).
        #Mbh = np.array([3.0, 5.0, 8.0, 10.0, 15.0]) # Solar masses
        #Rd = np.array([50.0, 50.0, 60.0, 90.0, 120.0]) # km

        # Use linear interpolation of the above values to find the disc radius in km
        #Rdisc = np.interp(self._Mbh, Mbh, Rd)
        # MAY NEED A SPECIFIC EXCEPTION FOR BH MASSES ABOVE 15; THE DATA
        # ARE RISING SHARPLY BUT INTERPOLATION JUST USES THE MAX VALUE

        # Disc compactness parameter (equation 3, Fernandez et al. 2020).
        #Cd = self._Mbh / 5.0 * 50.0 / Rdisc

        # Linear fit to the data in Table 2 of Fernandez et al. (2020), replicating the relation in their Figure 2.
        #f_ej = -0.19468593 * Cd + 0.30597772 # Fraction of the disc that is ejected in the disc wind.
        ##########

        # Updated ejected disc fraction from Equation 12 in Raaijmakers et al. (2021)
        # xi1 range is 0.04 - 0.32
        # xi2 range is 0.14 - 0.44.
        xi1 = 0.18
        xi2 = 0.29
        # Here I have split the difference of the lower and upper bounds.

        f_ej = xi1 + (xi2 - xi1) / (1.0 + np.exp(1.5 * (1.0 / self._q - 3.0)))

        Mwind_thermal = Mdisc * f_ej
        # Thermal disc wind outflow velocity (e.g. Kasen et al. 2015; Fernandez et al. 2020).
        Vthermal = 0.034 * ckm # Mean value of Table 2 in Fernandez et al. (2020).

        #################### SPLIT THERMAL WIND INTO BLUE AND PURPLE
        # Calculate the blue mass fraction of the thermally driven wind using the
        # observed relation between blue mass fraction and disc mass in
        # Fernandez et al. (2020), Table 2.

        # Parameters here are the result of a first-order polynomial fit.
        f_blue = 0.20199 * np.log10(Mdisc) + 1.12629
        Mblue_wind = Mwind_thermal * f_blue
        Mpurple_wind = Mwind_thermal * (1. - f_blue)

        # Now calculate the enhancement of the blue mass due to the spin of the post-merger BH. CURRENTLY UNUSED.
        # This effect is due to increased neutrino irradiation as the greater spin allows material
        # to sink deeper into the potential well of the BH.
        # The trend comes from Table 1 from Fernandez et al. (2015). Some fraction of material becomes
        # blue polar material with spins of 0.8 or more. Normalised to the 0.8 spin models of Fernandez et al. (2020).

        # chi_array = np.array([0.0, 0.4, 0.8, 0.95])
        # irradiated_array = np.array([0.0, 0.0, 0.01, 0.03])
        # irradiated = np.linterp(self._RemChi, chi_array, irradiated_array)

        # Normalisation to 0.8 spin models means that 0.01 of the purple mass had been transferred
        # already and must be returned for lower spins. More is transferred for higher spins.
        # Mblue_wind -= Mpurple_wind * (1.0 - (0.99 + irradiated))
        # Mpurple_wind += Mpurple_wind * (1.0 - (0.99 + irradiated))
        ####################

        # The magnetically driven wind will contribute some ejecta fraction as well.
        # Fernandez et al. (2019) show that it has comparable mass to the thermal wind.
        # The magnitude depends on the magnetic field geometry (e.g. Christie et al. (2019)).
        self._fMag = kwargs[self.key('fMag')]   # Magnetic field geometry ignorance parameter.
        Mwind_magnetic = Mwind_thermal * self._fMag
        Vmagnetic = max(0.22 * self._fMag * ckm, Vthermal) # 0.22c is the peak of the faster bimodal in Fernandez et al. (2019).

        ##########
        # Mixed wind properties (not used by default).
        # r-process module needs individual component kappas and masses.
        # Diffusion module needs the layer and layers above (averaged).
        # NB kappas are not loaded into this module by default so the code below will fail unless they're added.

        #Mwind = Mwind_thermal + Mwind_magnetic

        #Vejecta_mean = (Mwind_magnetic * Vmagnetic + Mwind_thermal * Vthermal + Mdyn * Vdyn) / (Mwind + Mdyn)

        #kappa_thermal_diffusion = (Mwind_thermal * self._kappa_purple + Mwind_magnetic * self._kappa_red) / Mwind
        ##########

        return {self.key('mejecta_magnetic'): Mwind_magnetic,
                self.key('mejecta_thermal'): Mwind_thermal,
                self.key('mejecta_thermal_blue'): Mblue_wind,
                self.key('mejecta_thermal_purple'): Mpurple_wind,
                self.key('mejecta_dyn'): Mdyn,
                self.key('vejecta_magnetic'): Vmagnetic,
                self.key('vejecta_thermal'): Vthermal,
                self.key('vejecta_dyn'): Vdyn,
                self.key('M1'): self._Mbh,
                self.key('M2'): self._Mns,
                self.key('Mrem'): Mrem,
                self.key('Mdisc'): Mdisc,
                self.key('f_ej'): f_ej,
                self.key('radius_ns'): self._radius_ns,
                self.key('Lambda_ns'): self._Lambda_ns,
                self.key('Mchirp'): self._mchirp,
                self.key('q'): self._q
                }
