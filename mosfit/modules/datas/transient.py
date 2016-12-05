import numpy as np
from mosfit.modules.module import Module
from mosfit.utils import is_number, listify

CLASS_NAME = 'Transient'


class Transient(Module):
    """Structure to store transient data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._keys = kwargs.get('keys', '')
        self._data_determined_parameters = []

    def process(self, **kwargs):
        return self._data

    def set_data(self,
                 all_data,
                 req_key_values={},
                 subtract_minimum_keys=[],
                 smooth_times=-1):
        self._all_data = all_data
        self._data = {}
        if not self._all_data:
            return
        name = list(self._all_data.keys())[0]
        for key in self._keys:
            if key not in self._all_data[name]:
                continue
            subdata = self._all_data[name][key]
            subkeys = self._keys[key]
            req_subkeys = [
                x for x in subkeys
                if not isinstance(subkeys, dict) or 'required' in listify(
                    subkeys[x])
            ]
            num_subkeys = [
                x for x in subkeys if 'numeric' in listify(subkeys[x])
            ]
            boo_subkeys = [
                x for x in subkeys if 'boolean' in listify(subkeys[x])
            ]
            exc_subkeys = [
                x for x in subkeys if 'exclude' in listify(subkeys[x])
            ]
            # Only include data that contains all subkeys
            for entry in subdata:
                if any([x not in entry for x in req_subkeys]):
                    continue
                if any([x in entry for x in exc_subkeys]):
                    continue
                if any([
                        x in entry and
                    (not is_number(entry[x]) or np.isnan(float(entry[x])))
                        for x in num_subkeys
                ]):
                    continue
                skip_key = False
                for qkey in req_key_values:
                    if (qkey in entry and
                            entry[qkey] not in req_key_values[qkey]):
                        skip_key = True
                        break
                if skip_key:
                    continue
                for x in subkeys:
                    falseval = False if x in boo_subkeys else ''
                    if x == 'value':
                        self._data[key] = entry.get(x, falseval)
                    else:
                        self._data.setdefault(
                            x + 's', []).append(entry.get(x, falseval))

        for key in self._data.copy():
            if isinstance(self._data[key], list):
                if not any(is_number(x) for x in self._data[key]):
                    continue
                self._data[key] = [
                    float(x) if is_number(x) else x for x in self._data[key]
                ]
                num_values = [
                    x for x in self._data[key] if isinstance(x, float)
                ]
                self._data['min_' + key] = min(num_values)
                self._data['max_' + key] = max(num_values)
            else:
                if is_number(self._data[key]):
                    self._data[key] = float(self._data[key])
                    self._data_determined_parameters.append(key)

        if 'times' in self._data and smooth_times >= 0:
            obs = list(
                zip(*(self._data['systems'], self._data['instruments'],
                      self._data['bands'])))

            uniqueobs = []
            for o in obs:
                if o not in uniqueobs:
                    uniqueobs.append(o)

            mint, maxt = min(self._data['times']), max(self._data['times'])
            alltimes = list(
                sorted(
                    set([x for x in self._data['times']] + (
                        np.linspace(mint, maxt, smooth_times)
                        if smooth_times > 0 else []))))

            obslist = list(
                zip(*(self._data['times'], self._data['systems'], self._data[
                    'instruments'], self._data['bands'], self._data[
                        'magnitudes'], self._data['e_magnitudes'],
                      self._data['e_lower_magnitudes'], self._data[
                          'e_upper_magnitudes'], self._data['upperlimits'],
                      [True for x in range(len(self._data['times']))])))

            for t in alltimes:
                for o in uniqueobs:
                    newobs = (t, o[0], o[1], o[2], None, None, None, None,
                              False, False)
                    if newobs not in obslist:
                        obslist.append(newobs)

            obslist.sort(key=lambda x: x[0])

            (self._data['times'], self._data['systems'],
             self._data['instruments'], self._data['bands'],
             self._data['magnitudes'], self._data['e_magnitudes'],
             self._data['e_lower_magnitudes'],
             self._data['e_upper_magnitudes'], self._data['upperlimits'],
             self._data['observed']) = zip(*obslist)

        for qkey in subtract_minimum_keys:
            minv = self._data['min_' + qkey]
            self._data[qkey] = [x - minv for x in self._data[qkey]]

    def get_data_determined_parameters(self):
        return self._data_determined_parameters
