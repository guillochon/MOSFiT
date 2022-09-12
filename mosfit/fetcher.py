"""A class for downloading event data from Open Catalogs."""
import codecs
import json
import os
import re
import shutil
import webbrowser
from collections import OrderedDict
from difflib import get_close_matches

from astrocats.catalog.utils import is_number
from six import string_types

from mosfit.printer import Printer
from mosfit.utils import get_url_file_handle, listify, open_atomic


class Fetcher(object):
    """Downloads data from the Open Catalogs."""

    def __init__(self,
                 test=False,
                 open_in_browser=False,
                 printer=None,
                 **kwargs):
        """Initialize class."""
        self._test = test
        self._printer = Printer() if printer is None else printer
        self._open_in_browser = open_in_browser

        self._names_downloaded = False
        self._names = OrderedDict()
        self._excluded_catalogs = []
        
        astrocatalogs_github_reponames = [
             'sne-1990-1999',
             'sne-2000-2004',
             'sne-2005-2009',
             'sne-2010-2014',
             'sne-2015-2019',
             'sne-2020-2024',
             'tde-1980-2025',
             'tne-2000-2029',
             'tde-external',
        ]
        self._catalogs = OrderedDict([
            (name, {'json': 'https://raw.githubusercontent.com/astrocatalogs/%s/master/' % name, 'web': None})
            for name in astrocatalogs_github_reponames
        ])

    def add_excluded_catalogs(self, catalogs):
        """Add catalog name(s) to list of catalogs that will be excluded."""
        if not isinstance(catalogs, list) or isinstance(
                catalogs, string_types):
            catalogs = listify(catalogs)
        self._excluded_catalogs.extend([x.upper() for x in catalogs])

    def fetch(self, event_list, offline=False, prefer_cache=False,
                cache_path=''):
        """Fetch a list of events from the open catalogs."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        prt = self._printer

        self._cache_path = cache_path

        levent_list = listify(event_list)
        events = [None for x in levent_list]

        catalogs = OrderedDict([(x, self._catalogs[x]) for x in self._catalogs
                                if x not in self._excluded_catalogs])

        for ei, event in enumerate(levent_list):
            if not event:
                continue
            events[ei] = OrderedDict()
            path = ''
            # If the event name ends in .json, assume event is a path.
            if event.endswith('.json'):
                path = event
                events[ei]['name'] = event.replace('.json', '').split('/')[-1]

            # If not (or the file doesn't exist), download from an open
            # catalog.

            name_dir_path = dir_path
            if self._cache_path:
                name_dir_path = self._cache_path

            if not path or not os.path.exists(path):
                names_paths = [
                    os.path.join(name_dir_path, 'cache', x + '.names.min.json')
                    for x in catalogs
                ]
                input_name = event.replace('.json', '')
                for ci, catalog in enumerate(catalogs):
                    events[ei]['name'] = input_name
                    events[ei]['catalog'] = catalog

                    urlname = events[ei]['name'] + '.json'
                    name_path = os.path.join(name_dir_path, 'cache', urlname)
                    url = catalogs[events[ei]['catalog']]['json'] + urlname

                    if offline or (prefer_cache and os.path.exists(name_path)):
                        prt.message('cached_event',
                                    [events[ei]['name'], events[ei]['catalog']])
                    else:
                        prt.message('dling_event',
                                    [events[ei]['name'], events[ei]['catalog']])
                        try:
                            response = get_url_file_handle(url, timeout=10)
                        except Exception:
                            prt.message('cant_dl_event', [url], warning=True)
                            continue
                        else:
                            with open_atomic(name_path, 'wb') as f:
                                shutil.copyfileobj(response, f)
                    path = name_path

            if os.path.exists(path):
                events[ei]['path'] = path
                if self._open_in_browser and catalogs[events[ei]['catalog']]['web'] is not None:
                    webbrowser.open(catalogs[events[ei]['catalog']]['web'] +
                                    events[ei]['name'])
                prt.message('event_file', [path], wrapped=True)
            else:
                prt.message('no_data',
                            [events[ei]['name'], '/'.join(catalogs.keys())])
                if offline:
                    prt.message('omit_offline')
                raise RuntimeError

        return events

    def load_data(self, event):
        """Return data from specified path."""
        if not os.path.exists(event['path']):
            return None
        with codecs.open(event['path'], 'r', encoding='utf-8') as f:
            return json.load(f, object_pairs_hook=OrderedDict)
