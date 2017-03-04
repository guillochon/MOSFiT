from mosfit.modules.module import Module

# Important: Only define one `Module` class per file.


class Engine(Module):
    """Generic engine module.
    """

    def process(self, **kwargs):
        return {}

    def add_to_existing_lums(self, new_lums, **kwargs):
        # Add on to any existing luminosity
        old_lums = kwargs.get('luminosities', None)
        if old_lums is not None:
            new_lums = [x + y for x, y in zip(old_lums, new_lums)]
        return new_lums
