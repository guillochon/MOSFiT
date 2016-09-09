import importlib
import json
import logging
from collections import Iterable, OrderedDict
from logging import DEBUG, INFO, WARNING

import emcee
import numpy as np


class Model:
    def __init__(self, parameters=[], path='example_model.json'):
        self._path = path
        with open(path, 'r') as f:
            self._model_dict = json.loads(f.read())
        self._parameters = parameters
        self._log = logging.getLogger()

        # Load the call tree for the model. Work our way in reverse from the
        # observables, first constructing a tree for each observable and then
        # combining trees.
        trees = {}
        self.construct_trees(self._model_dict, trees, kind='observable')

        unsorted_call_stack = {}
        max_depth_all = -1
        for tag in self._model_dict:
            if self._model_dict[tag]['kind'] == 'observable':
                max_depth = 0
            else:
                max_depth = -1
                for tag2 in trees:
                    depth = self.get_max_depth(tag, trees[tag2])
                    if depth > max_depth:
                        max_depth = depth
                    if depth > max_depth_all:
                        max_depth_all = depth
            new_entry = self._model_dict[tag].copy()
            if 'children' in new_entry:
                del(new_entry['children'])
            new_entry['depth'] = max_depth
            unsorted_call_stack[tag] = new_entry
        # print(unsorted_call_stack)

        call_stack = OrderedDict()
        for depth in range(max_depth_all, -1, -1):
            for task in unsorted_call_stack:
                if unsorted_call_stack[task]['depth'] == depth:
                    call_stack[task] = unsorted_call_stack[task]

        print(call_stack)
        raise (SystemExit)

        # try:
        #     mod = importlib.import_module('.' + mod_name, package='astrocats')
        # except Exception as err:
        #     self._log.error("Import of specified module '{}' failed.".format(mod_name))

    def get_max_depth(self, tag, parent):
        for child in parent.get('children', []):
            if child == tag:
                return parent['children'][child]['depth']
            else:
                depth = self.get_max_depth(tag, parent['children'][child])
                return depth
        return parent['depth']

    def construct_trees(self, d, trees, kind='', name='', depth=0):
        for tag in d:
            entry = d[tag].copy()
            if entry['kind'] == kind or tag == name:
                entry['depth'] = depth
                trees[tag] = entry
                inputs = []
                if 'inputs' in entry:
                    if isinstance(entry['inputs'], str):
                        inputs = [entry['inputs']]
                    else:
                        inputs = entry['inputs']
                for inp in inputs:
                    children = {}
                    self.construct_trees(
                        d, children, name=inp, depth=depth + 1)
                    trees[tag].setdefault('children', {})
                    trees[tag]['children'].update(children)

    def flatten_tree(self, l):
        for el in l:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                for sub in self.flatten_tree(el):
                    yield sub
            else:
                yield el

    def lnprob(self, x, ivar):
        parameters = {}
        for par in self._parameters:
            parameters[par.name] = (
                ivar *
                (par['max_value'] - par['min_value']) + par['min_value'])
        return -0.5 * np.sum(ivar * x**2)

    def fit_data(self, data, plot_times=[]):
        ndim, nwalkers = 10, 100
        ivar = 1. / np.random.rand(ndim)
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.lnprob, args=[ivar])
        sampler.run_mcmc(p0, 1000)
