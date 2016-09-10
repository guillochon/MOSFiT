class Module:
    def __init__(self, name, **kwargs):
        self._name = name

    def process(self, **kwargs):
        return {}

    def request(self, request):
        return []

    def handle_requests(self, **kwargs):
        pass
