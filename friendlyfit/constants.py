import astropy.units as u
import numpy as np

LOG2 = np.log(2.0)
LIKELIHOOD_FLOOR = -np.inf
FOUR_PI = 4.0 * np.pi
MAG_FAC = 100.0**0.2
AB_OFFSET = -48.60
MPC_CGS = (1.0 * u.Mpc).cgs.value
