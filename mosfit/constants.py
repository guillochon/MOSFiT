"""List of numerical constants, faster than using astropy with every call."""

from decimal import Decimal

import astropy.constants as c
import astropy.units as u
import numpy as np

LIKELIHOOD_FLOOR = -np.inf
LOCAL_LIKELIHOOD_FLOOR = -1.0e8

ANG_CGS = u.Angstrom.cgs.scale
AU_CGS = u.au.cgs.scale
C_CGS = c.c.cgs.value
H_CGS = c.h.cgs.value
DAY_CGS = u.day.cgs.scale
FOE = 1.0e51
FOUR_PI = 4.0 * np.pi
SQRT_2_PI = np.sqrt(2.0 * np.pi)
IPI = 1.0 / np.pi
KM_CGS = u.km.cgs.scale
M_SUN_CGS = c.M_sun.cgs.value
M_P_CGS = c.m_p.cgs.value
MEV_CGS = u.MeV.cgs.scale
MAG_FAC = 2.5
MPC_CGS = u.Mpc.cgs.scale
PC_CGS = u.pc.cgs.scale

# Absolute-flux reference distance (10 pc) used when MOSFiT reports rest-frame
# SEDs as flux densities, and the CGS value of one nanojansky. Together these
# convert ``seds`` (erg/s/Angstrom) into the nJy-at-10-pc convention that
# external simulators such as LightCurveLynx expect.
TEN_PC_CGS = 10.0 * PC_CGS
NJY_CGS = 1.0e-32  # erg / s / cm^2 / Hz

# ``all_band_indices`` sentinel for bolometric / intrinsic luminosity rows
# (catalog ``luminosity`` field); distinct from radio ``band_index == -1``.
BOL_BAND_INDEX = -2

# Reserved photometry ``band`` for absolute bolometric magnitude (``mbol``)
# catalog rows; avoids colliding with real filter names.
BOL_MAG_BAND_LABEL = 'Bolometric'

# Calibration: absolute bolometric mag of the Sun vs ``astropy.constants.L_sun``
# bolometric luminosity (erg/s).
MBOL_SUN_MAG = 4.74
LBOL_SUN_CGS = c.L_sun.cgs.value

KS_DAYS = float(Decimal('1000') / Decimal(DAY_CGS))
H_C_CGS = H_CGS * C_CGS
H_C_ANG_CGS = H_C_CGS / ANG_CGS

G_CGS = c.G.cgs.value
