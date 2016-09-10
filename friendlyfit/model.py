import importlib
import json
import logging
from collections import OrderedDict

import emcee
import numpy as np


class Model:
    def __init__(self,
                 parameter_path='parameters.json',
                 model_path='example_model.json'):
        self._model_path = model_path
        with open(model_path, 'r') as f:
            self._model = json.loads(f.read())
        with open(parameter_path, 'r') as f:
            self._parameters = json.loads(f.read())
        self._num_free_parameters = len(
            [x for x in self._parameters
             if ('min_value' in self._parameters[x] and 'max_value' in
                 self._parameters[x])])
        self._log = logging.getLogger()
        self._modules = {}

        # Load the call tree for the model. Work our way in reverse from the
        # observables, first constructing a tree for each observable and then
        # combining trees.
        root_kinds = ['observable', 'objective']

        trees = {}
        self.construct_trees(self._model, trees, kinds=root_kinds)

        unsorted_call_stack = {}
        self._max_depth_all = -1
        for tag in self._model:
            if self._model[tag]['kind'] in root_kinds:
                max_depth = 0
            else:
                max_depth = -1
                for tag2 in trees:
                    depth = self.get_max_depth(tag, trees[tag2], max_depth)
                    if depth > max_depth:
                        max_depth = depth
                    if depth > self._max_depth_all:
                        self._max_depth_all = depth
            new_entry = self._model[tag].copy()
            if 'children' in new_entry:
                del (new_entry['children'])
            new_entry['depth'] = max_depth
            unsorted_call_stack[tag] = new_entry
        # print(unsorted_call_stack)

        # Currently just have one call stack for all products, can be wasteful
        # if only using some products.
        self._call_stack = OrderedDict()
        for depth in range(self._max_depth_all, -1, -1):
            for task in unsorted_call_stack:
                if unsorted_call_stack[task]['depth'] == depth:
                    self._call_stack[task] = unsorted_call_stack[task]

        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if 'class' in cur_task:
                class_name = cur_task['class']
            else:
                class_name = task
            mod = importlib.import_module(
                '.' + 'modules.' + cur_task['kind'] + 's.' + class_name,
                package='friendlyfit')
            mod_class = getattr(mod, mod.CLASS_NAME)
            if cur_task['kind'] == 'parameter' and task in self._parameters:
                cur_task.update(self._parameters[task])
            self._modules[task] = mod_class(name=task, **cur_task)
            if 'requests' in cur_task:
                requests = {}
                parent = ''
                for par in self._call_stack:
                    if par == cur_task['inputs']:
                        parent = par
                if not parent:
                    self._log.error("Couldn't find parent task!")
                    raise (RuntimeError)
                for req in cur_task['requests']:
                    requests.setdefault(
                        req, []).append(self._modules[task].request(req))
                self._modules[parent].handle_requests(**requests)

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

    def lnprob(self, x, ivar):
        inputs = {}
        outputs = {}
        pos = 0
        cur_depth = self._max_depth_all
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['depth'] != cur_depth:
                inputs = outputs
                outputs = {}
            cur_depth = cur_task['depth']
            if (cur_task['kind'] == 'parameter' and 'min_value' in cur_task and
                    'max_value' in cur_task):
                inputs.update({'fraction': x[pos]})
                pos = pos + 1
            new_outs = self._modules[task].process(**inputs)
            outputs.update(new_outs)

            if self._call_stack[task]['kind'] == 'objective':
                return outputs['value']

        # return -0.5 * np.sum(ivar * x**2)

    def fit_data(self, data, plot_points=[]):
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['kind'] == 'data':
                self._modules[task].set_data(data)

        ndim, nwalkers = self._num_free_parameters, 100
        ivar = 1. / np.random.rand(ndim)
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.lnprob, args=[ivar])
        sampler.run_mcmc(p0, 10)

        return (p0, p0)
