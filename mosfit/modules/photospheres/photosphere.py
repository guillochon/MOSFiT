import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.modules.module import Module

CLASS_NAME = 'photosphere'


class photosphere(Module):
    """Template class for photosphere Modules.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        return {}
