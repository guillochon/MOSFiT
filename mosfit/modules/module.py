"""Definitions for the ``Module`` class."""
import json
from collections import OrderedDict

import numpy as np

from mosfit.printer import Printer


class Module(object):
    """Base ``Module`` class."""

    _REFERENCES = []

    def __init__(self, name, model, **kwargs):
        """Initialize module.

        This is where expensive calculations that only need to be evaluated
        once should be located.
        """
        self._name = name
        self._log = False
        self._model = model
        self._pool = model.pool()
        self._preprocessed = False
        self._wants_dense = False
        self._provide_dense = False
        self._replacements = OrderedDict()
        if not model.printer():
            self._printer = Printer()
        else:
            self._printer = model.printer()

    def __repr__(self):
        """Return a string representation of self."""
        return json.dumps(self.__dict__)

    def process(self, **kwargs):
        """Process module, should always return a dictionary."""
        return OrderedDict()

    def reset_preprocessed(self, exceptions):
        """Reset preprocessed flag."""
        if self._name not in exceptions:
            self._preprocessed = False

    def send_request(self, request):
        """Send a request."""
        return []

    def name(self):
        """Return own name."""
        return self._name

    def receive_requests(self, **requests):
        """Receive requests from other ``Module`` objects."""
        pass

    def set_event_name(self, event_name):
        """Set the name of the event being modeled."""
        self._event_name = event_name

    def set_attributes(self, task):
        """Set key replacement dictionary."""
        self._replacements = task.get('replacements', OrderedDict())
        if 'wants_dense' in task:
            self._wants_dense = task['wants_dense']

    def get_bibcode(self):
        """Return any bibcodes associated with the present ``Module``."""
        return []

    def dense_key(self, key):
        """Manipulate output keys conditionally."""
        new_key = self.key(key)
        if self._provide_dense and not key.startswith('dense_'):
            return 'dense_' + new_key
        return new_key

    def key(self, key):
        """Substitute user-defined replacement key names for local names."""
        new_key = key
        for rep in self._replacements:
            new_key = new_key.replace(rep, self._replacements[rep])
        return new_key

    def prepare_input(self, key, **kwargs):
        """Prepare keys conditionally."""
        if key not in kwargs:
            if 'dense_' + key in kwargs:
                kwargs[key] = np.take(
                    np.array(kwargs['dense_' + key]),
                    np.array(kwargs['dense_indices']))
            else:
                raise RuntimeError(
                    'Expecting `dense_` version of `{}` to exist before '
                    'calling `{}` module.'.format(key, self._name))
        return kwargs
