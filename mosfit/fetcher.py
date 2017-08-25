"""A class for downloading event data from Open Catalogs."""
import json
import os
import re
import shutil
import webbrowser
from collections import OrderedDict
from difflib import get_close_matches

from astrocats.catalog.utils import is_number

from mosfit.utils import get_url_file_handle, listify, open_atomic


class Fetcher(object):
    """Downloads data from the Open Catalogs."""

    def __init__(self, fitter, **kwargs):
        """Initialize class."""
        self._fitter = fitter
        self._printer = fitter._printer

    def fetch(self, event_list):
        """Fetch a list of events from the open catalogs."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        prt = self._printer
        offline = self._fitter._offline

        levent_list = listify(event_list)
        events = [None for x in levent_list]

        for ei, event in enumerate(levent_list):
            if not event:
                continue
            events[ei] = OrderedDict()
            path = ''
            # If the event name ends in .json, assume event is a path.
            if event.endswith('.json'):
                path = event
                events[ei]['name'] = event.replace('.json',
                                                   '').split('/')[-1]

            # If not (or the file doesn't exist), download from an open
            # catalog.
            if not path or not os.path.exists(path):
                names_paths = [
                    os.path.join(dir_path, 'cache', x +
                                 '.names.min.json') for x in
                    self._fitter._catalogs]
                input_name = event.replace('.json', '')
                if offline:
                    prt.message('event_interp', [input_name])
                else:
                    prt.message('dling_aliases', [input_name])
                    for ci, catalog in enumerate(self._fitter._catalogs):
                        try:
                            response = get_url_file_handle(
                                self._fitter._catalogs[catalog]['json'] +
                                '/names.min.json',
                                timeout=10)
                        except Exception:
                            prt.message('cant_dl_names'
                                        [catalog], warning=True)
                            raise
                        else:
                            with open_atomic(
                                    names_paths[ci], 'wb') as f:
                                shutil.copyfileobj(response, f)
                names = OrderedDict()
                for ci, catalog in enumerate(self._fitter._catalogs):
                    if os.path.exists(names_paths[ci]):
                        with open(names_paths[ci], 'r') as f:
                            names[catalog] = json.load(
                                f, object_pairs_hook=OrderedDict)
                    else:
                        prt.message('cant_read_names', [catalog],
                                    warning=True)
                        if offline:
                            prt.message('omit_offline')
                        raise RuntimeError

                    if input_name in names[catalog]:
                        events[ei]['name'] = input_name
                        events[ei]['catalog'] = catalog
                    else:
                        for name in names[catalog]:
                            if (input_name in names[catalog][name] or
                                    'SN' + input_name in
                                    names[catalog][name]):
                                events[ei]['name'] = name
                                events[ei]['catalog'] = catalog
                                break

                if not events[ei].get('name', None):
                    for ci, catalog in enumerate(self._fitter._catalogs):
                        namekeys = []
                        for name in names[catalog]:
                            namekeys.extend(names[catalog][name])
                        namekeys = list(sorted(set(namekeys)))
                        matches = get_close_matches(
                            event, namekeys, n=5, cutoff=0.8)
                        # matches = []
                        if len(matches) < 5 and is_number(event[0]):
                            prt.message('pef_ext_search')
                            snprefixes = set(('SN19', 'SN20'))
                            for name in names[catalog]:
                                ind = re.search("\d", name)
                                if ind and ind.start() > 0:
                                    snprefixes.add(name[:ind.start()])
                            snprefixes = list(sorted(snprefixes))
                            for prefix in snprefixes:
                                testname = prefix + event
                                new_matches = get_close_matches(
                                    testname, namekeys, cutoff=0.95,
                                    n=1)
                                if (len(new_matches) and
                                        new_matches[0] not in matches):
                                    matches.append(new_matches[0])
                                if len(matches) == 5:
                                    break
                        if len(matches):
                            if self._fitter._test:
                                response = matches[0]
                            else:
                                response = prt.prompt(
                                    'no_exact_match',
                                    kind='select',
                                    options=matches,
                                    none_string=(
                                        'None of the above, ' +
                                        ('skip this event.' if
                                         ci == len(self._fitter._catalogs) - 1
                                         else
                                         'try the next catalog.')))
                            if response:
                                for name in names[catalog]:
                                    if response in names[
                                            catalog][name]:
                                        events[ei]['name'] = name
                                        events[ei]['catalog'] = catalog
                                        break
                                if events[ei]['name']:
                                    break

                if not events[ei].get('name', None):
                    prt.message('no_event_by_name')
                    events[ei]['name'] = input_name
                    continue
                urlname = events[ei]['name'] + '.json'
                name_path = os.path.join(dir_path, 'cache', urlname)

                if offline:
                    prt.message('cached_event', [
                        events[ei]['name'], events[ei]['catalog']])
                else:
                    prt.message('dling_event', [
                        events[ei]['name'], events[ei]['catalog']])
                    try:
                        response = get_url_file_handle(
                            self._fitter._catalogs[events[ei]['catalog']][
                                'json'] + '/json/' + urlname,
                            timeout=10)
                    except Exception:
                        prt.message('cant_dl_event', [
                            events[ei]['name']], warning=True)
                    else:
                        with open_atomic(name_path, 'wb') as f:
                            shutil.copyfileobj(response, f)
                path = name_path

            if os.path.exists(path):
                if self._fitter._open_in_browser:
                    webbrowser.open(
                        self._fitter._catalogs[events[ei]['catalog']]['web'] +
                        events[ei]['name'])
                with open(path, 'r') as f:
                    events[ei]['data'] = json.load(
                        f, object_pairs_hook=OrderedDict)
                prt.message('event_file', [path], wrapped=True)
            else:
                prt.message('no_data', [
                    events[ei]['name'],
                    '/'.join(self._fitter._catalogs.keys())])
                if offline:
                    prt.message('omit_offline')
                raise RuntimeError

        return events
