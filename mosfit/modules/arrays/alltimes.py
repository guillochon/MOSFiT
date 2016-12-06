from mosfit.modules.module import Module

CLASS_NAME = 'AllTimes'


class AllTimes(Module):
    """This class converts the observed times to rest-frame times.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, **kwargs):
        if (kwargs.get('root', 'output') == 'output' and
                'extra_times' in kwargs):
            obslist = (
                list(
                    zip(*(kwargs['times'], kwargs['systems'], kwargs[
                        'instruments'], kwargs['bands'],
                          [True for x in range(len(kwargs['times']))]))) +
                list(
                    zip(*(kwargs['extra_times'], kwargs['extra_systems'],
                          kwargs['extra_instruments'], kwargs['extra_bands'],
                          [False for x in range(len(kwargs['extra_times']))])))
            )
            obslist.sort()

            (self._times, self._systems, self._instruments, self._bands,
             self._observed) = zip(*obslist)
        else:
            self._times = kwargs['times']
            self._systems = kwargs['systems']
            self._instruments = kwargs['instruments']
            self._bands = kwargs['bands']
            self._observed = [True for x in kwargs['times']]

        outputs = {}
        outputs['all_times'] = self._times
        outputs['all_systems'] = self._systems
        outputs['all_instruments'] = self._instruments
        outputs['all_bands'] = self._bands
        outputs['observed'] = self._observed
        return outputs
