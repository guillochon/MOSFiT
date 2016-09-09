from ..module import Module

CLASS_NAME = 'Parameter'


class Parameter(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ivar = kwargs['fraction']
        #
        # value = (ivar *
        #                    (cur_par['max_value'] - cur_par['min_value']
        #                     ) + cur_par['min_value'])
