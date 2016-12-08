import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.modules.module import Module

CLASS_NAME = 'energetic'


class energetic(Module):
    """Template class for energy/velocity conversions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        return {}
