import importlib
import json
import logging
import sys
from collections import OrderedDict

import emcee
import numpy as np
from emcee.utils import MPIPool
from tqdm import tqdm


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
        self._bands = []

        # Load the call tree for the model. Work our way in reverse from the
        # observables, first constructing a tree for each observable and then
        # combining trees.
        root_kinds = ['objective']

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
                if cur_task['class'] == 'band':
                    self._bands.append(cur_task['band'])
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
                inputs = []
                if 'inputs' in cur_task:
                    if isinstance(cur_task['inputs'], str):
                        inputs = [cur_task['inputs']]
                    else:
                        inputs = cur_task['inputs']
                for i, inp in enumerate(inputs):
                    requests = {}
                    parent = ''
                    for par in self._call_stack:
                        if par == inp:
                            parent = par
                    if not parent:
                        self._log.error("Couldn't find parent task!")
                        raise (RuntimeError)
                    reqs = cur_task['requests'][i]
                    for req in reqs:
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

    def run_stack(self, x, root='objective'):
        inputs = {}
        outputs = {}
        pos = 0
        cur_depth = self._max_depth_all
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['depth'] != cur_depth:
                inputs = outputs
            cur_depth = cur_task['depth']
            if (cur_task['kind'] == 'parameter' and 'min_value' in cur_task and
                    'max_value' in cur_task):
                inputs.update({'fraction': x[pos]})
                pos = pos + 1
            # if root == 'observable':
            #     inputs['times']:
            new_outs = self._modules[task].process(**inputs)
            outputs.update(new_outs)

            if self._call_stack[task]['kind'] == root:
                if root == 'objective':
                    if min(x) < 0.0 or max(x) > 1.0:
                        return -np.inf
                    return outputs['value']
        # Should not reach here, should always have an output
        self._log.error('run_stack should have produced an output!')
        raise RuntimeError

    def fit_data(self, data, plot_points=[], iterations=10, num_walkers=100):
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['kind'] == 'data':
                if cur_task['class'] == 'photometry':
                    self._modules[task].set_data(data, self._bands)
                elif cur_task['class'] == 'quantity':
                    self._modules[task].set_data(data)

        ndim, nwalkers = self._num_free_parameters, num_walkers
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]

        pool = MPIPool(loadbalance=True)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.run_stack, args=['objective'], pool=pool)
        for result in tqdm(
                sampler.sample(
                    p0, iterations=iterations), total=iterations):
            # pass
            tqdm.write(str(max(result[1])))
        pool.close()

        return (p0, p0)
