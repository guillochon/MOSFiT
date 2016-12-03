class Module:
    def __init__(self, name, pool, **kwargs):
        self._name = name
        self._log = False
        self._pool = pool

    def process(self, **kwargs):
        return {}

    def request(self, request):
        return []

    def name(self):
        return self._name

    def is_log(self):
        return self._log

    def handle_requests(self, **kwargs):
        pass

    def set_event_name(self, event_name):
        self._event_name = event_name
