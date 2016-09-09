import importlib
import json
import logging
from collections import Iterable, OrderedDict

import emcee
import numpy as np


class Model:
    def __init__(self, parameters=[], path='example_model.json'):
        self._path = path
        with open(path, 'r') as f:
            self._model_dict = json.loads(f.read())
        self._parameters = parameters
        self._log = logging.getLogger()
        self._modules = {}

        # Load the call tree for the model. Work our way in reverse from the
        # observables, first constructing a tree for each observable and then
        # combining trees.
        root_kinds = ['observables', 'objective']

        trees = {}
        self.construct_trees(self._model_dict, trees, kinds=root_kinds)

        unsorted_call_stack = {}
        max_depth_all = -1
        for tag in self._model_dict:
            if self._model_dict[tag]['kind'] in root_kinds:
                max_depth = 0
            else:
                max_depth = -1
                for tag2 in trees:
                    depth = self.get_max_depth(tag, trees[tag2], max_depth)
                    if depth > max_depth:
                        max_depth = depth
                    if depth > max_depth_all:
                        max_depth_all = depth
            new_entry = self._model_dict[tag].copy()
            if 'children' in new_entry:
                del (new_entry['children'])
            new_entry['depth'] = max_depth
            unsorted_call_stack[tag] = new_entry
        # print(unsorted_call_stack)

        self._call_stack = OrderedDict()
        for depth in range(max_depth_all, -1, -1):
            for task in unsorted_call_stack:
                if unsorted_call_stack[task]['depth'] == depth:
                    self._call_stack[task] = unsorted_call_stack[task]

        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['kind'] in ['engine', 'observable', 'sed', 'transform'
                                    ]:
                if 'class' in cur_task:
                    class_name = cur_task['class']
                else:
                    class_name = task
                self._modules[task] = importlib.import_module(
                    '.' + 'modules.' + cur_task['kind'] + 's.' + class_name,
                    package='friendlyfit')

    def get_max_depth(self, tag, parent, max_depth):
        for child in parent.get('children', []):
            if child == tag:
                new_max = parent['children'][child]['depth']
                if new_max > max_depth:
                    max_depth = new_max
            else:
                new_max = self.get_max_depth(tag, parent['children'][child],
                                             max_depth)
                if new_max > max_depth:
                    max_depth = new_max
        return max_depth

    def construct_trees(self, d, trees, kinds=[], name='', depth=0):
        for tag in d:
            entry = d[tag].copy()
            if entry['kind'] in kinds or tag == name:
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

        return self.magdev(parameters)
        # return -0.5 * np.sum(ivar * x**2)

    # def magdev(self, parameters):
    #     for task in self._call_stack:

    def fit_data(self, data, plot_times=[]):
        ndim, nwalkers = 10, 100
        ivar = 1. / np.random.rand(ndim)
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.lnprob, args=[ivar])
        sampler.run_mcmc(p0, 1000)
