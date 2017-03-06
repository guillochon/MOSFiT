"""Definitions for the `Constraint` class."""
import numpy as np

from mosfit.modules.module import Module
from mosfit.utils import listify

# Important: Only define one ``Module`` class per file.

class Constraint(Module):
    def __init__(self, **kwargs):
        super(Constraint, self).__init__(**kwargs)
    def process(self, **kwargs):
        self._score_modifier = 0.0
        return {'score_modifier':self._score_modifier}
