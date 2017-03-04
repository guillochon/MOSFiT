"""Definitions for the `Module` class."""
from mosfit.printer import Printer


class Module(object):
    """Base `Module` class."""

    def __init__(self, name, pool, printer=None, **kwargs):
        self._name = name
        self._log = False
        self._pool = pool
        if not printer:
            self._printer = Printer()
        else:
            self._printer = printer
        self._printer = printer

    def process(self, **kwargs):
        return {}

    def send_request(self, request):
        return []

    def name(self):
        return self._name

    def is_log(self):
        return self._log

    def receive_requests(self, **kwargs):
        pass

    def set_event_name(self, event_name):
        self._event_name = event_name

    def get_bibcode(self):
        return ''
