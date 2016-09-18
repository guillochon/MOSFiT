class Module:
    def __init__(self, name, **kwargs):
        self._name = name

    def process(self, **kwargs):
        return {}

    def request(self, request):
        return []

    def name(self):
        return self._name

    def handle_requests(self, **kwargs):
        pass

    def set_event_name(self, event_name):
        self._event_name = event_name
