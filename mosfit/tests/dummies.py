"""Shared dummy pool/model/printer objects for unit tests."""


class DummyPool(object):
    size = 0
    comm = None

    def is_master(self):
        return True


class DummyPrinter(object):
    def message(self, *a, **k):
        return ''

    def text(self, *a, **k):
        return ''

    def prt(self, *a, **k):
        return ''


class DummyFitter(object):
    _event_name = 'dummy'
    _limiting_magnitude = None
    _prefer_cache = True
    _cuda = False


class DummyModel(object):
    def __init__(self):
        self._fitter = DummyFitter()

    def pool(self):
        return DummyPool()

    def printer(self):
        return DummyPrinter()
