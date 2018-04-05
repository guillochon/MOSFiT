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

KS_DAYS = float(Decimal('1000') / Decimal(DAY_CGS))
H_C_CGS = H_CGS * C_CGS
H_C_ANG_CGS = H_C_CGS / ANG_CGS
