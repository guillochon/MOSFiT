"""List of numerical constants, faster than using astropy with every call.
"""

import astropy.constants as c
import astropy.units as u
import numpy as np

LOG2 = np.log(2.0)
LIKELIHOOD_FLOOR = -np.inf
LOCAL_LIKELIHOOD_FLOOR = -1.0e5
FOUR_PI = 4.0 * np.pi
MAG_FAC = 100.0**0.2
AB_OFFSET = -48.60
MPC_CGS = (1.0 * u.Mpc).cgs.value
DAY_CGS = (1.0 * u.day).cgs.value
M_SUN_CGS = c.M_sun.cgs.value
KM_CGS = 1.0e5
C_CGS = c.c.cgs.value
