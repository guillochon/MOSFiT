"""Definitions for the `Model` class."""
import importlib
import inspect
import json
import os
from collections import OrderedDict
from copy import deepcopy
from difflib import get_close_matches
from math import isnan

import inflect
import numpy as np
from mosfit.constants import LOCAL_LIKELIHOOD_FLOOR
from mosfit.modules.module import Module
from mosfit.printer import Printer
from mosfit.utils import listify
# from scipy.optimize import differential_evolution
from scipy.optimize import minimize
# from scipy.optimize import basinhopping

from astrocats.catalog.quantity import QUANTITY
# from bayes_opt import BayesianOptimization


class Model(object):
    """Define a semi-analytical model to fit transients with."""

    MODEL_OUTPUT_DIR = 'products'
    MIN_WAVE_FRAC_DIFF = 0.1

    # class outClass(object):
    #     pass

    def __init__(self,
                 parameter_path='parameters.json',
                 model='',
                 data={},
                 wrap_length=100,
                 pool=None,
                 fitter=None,
                 print_trees=False):
        """Initialize `Model` object."""
        self._model_name = model
        self._pool = pool
        self._is_master = pool.is_master() if pool else False
        self._wrap_length = wrap_length
        self._fitter = fitter
        self._print_trees = print_trees
        self._inflect = inflect.engine()
        self._inflections = {}
        self._references = []

        if self._fitter:
            self._printer = self._fitter._printer
        else:
            self._printer = Printer(pool=pool, wrap_length=wrap_length)

        prt = self._printer

        self._dir_path = os.path.dirname(os.path.realpath(__file__))

        # Load suggested model associations for transient types.
        if os.path.isfile(os.path.join('models', 'types.json')):
            types_path = os.path.join('models', 'types.json')
        else:
            types_path = os.path.join(self._dir_path, 'models',
                                      'types.json')
        with open(types_path, 'r') as f:
            model_types = json.load(f, object_pairs_hook=OrderedDict)

        if not self._model_name:
            try:
                claimed_type = list(data.values())[0][
                    'claimedtype'][0][QUANTITY.VALUE]
            except Exception:
                prt.message('no_model_type', warning=True)
            else:
                type_options = model_types.get(claimed_type, [])
                if not type_options:
                    prt.message('no_model_for_type', warning=True)
                else:
                    if fitter._test:
                        self._model_name = type_options[0]
                    else:
                        self._model_name = self._printer.prompt(
                            'No model specified. Based on this transient\'s '
                            'claimed type of `{}`, the following models are '
                            'suggested for fitting this transient:'
                            .format(claimed_type),
                            kind='select',
                            options=type_options,
                            message=False,
                            none_string=('None of the above, skip this '
                                         'transient.'))

        if not self._model_name:
            return

        # Load the basic model file.
        if os.path.isfile(os.path.join('models', 'model.json')):
            basic_model_path = os.path.join('models', 'model.json')
        else:
            basic_model_path = os.path.join(self._dir_path, 'models',
                                            'model.json')

        with open(basic_model_path, 'r') as f:
            self._model = json.load(f, object_pairs_hook=OrderedDict)

        # Load the model file.
        model = self._model_name
        model_dir = self._model_name

        if '.json' in self._model_name:
            model_dir = self._model_name.split('.json')[0]
        else:
            model = self._model_name + '.json'

        if os.path.isfile(model):
            model_path = model
        else:
            # Look in local hierarchy first
            if os.path.isfile(os.path.join('models', model_dir, model)):
                model_path = os.path.join('models', model_dir, model)
            else:
                model_path = os.path.join(self._dir_path, 'models', model_dir,
                                          model)

        with open(model_path, 'r') as f:
            self._model.update(json.load(f, object_pairs_hook=OrderedDict))

        # with open(os.path.join(
        #         self.MODEL_OUTPUT_DIR,
        #         self._model_name + '.json'), 'w') as f:
        #     json.dump(self._model, f)

        # Load model parameter file.
        model_pp = os.path.join(
            self._dir_path, 'models', model_dir, 'parameters.json')

        pp = ''

        local_pp = (parameter_path if '/' in parameter_path
                    else os.path.join('models', model_dir, parameter_path))

        if os.path.isfile(local_pp):
            selected_pp = local_pp
        else:
            selected_pp = os.path.join(
                self._dir_path, 'models', model_dir, parameter_path)

        # First try user-specified path
        if parameter_path and os.path.isfile(parameter_path):
            pp = parameter_path
        # Then try directory we are running from
        elif os.path.isfile('parameters.json'):
            pp = 'parameters.json'
        # Then try the model directory, with the user-specified name
        elif os.path.isfile(selected_pp):
            pp = selected_pp
        # Finally try model folder
        elif os.path.isfile(model_pp):
            pp = model_pp
        else:
            raise ValueError('Could not find parameter file!')

        if self._is_master:
            prt.message('files', [basic_model_path, model_path, pp],
                        wrapped=False)

        with open(pp, 'r') as f:
            self._parameter_json = json.load(f, object_pairs_hook=OrderedDict)
        self._modules = OrderedDict()
        self._bands = []
        self._instruments = []

        # Load the call tree for the model. Work our way in reverse from the
        # observables, first constructing a tree for each observable and then
        # combining trees.
        root_kinds = ['output', 'objective']

        self._trees = OrderedDict()
        self._simple_trees = OrderedDict()
        self.construct_trees(
            self._model, self._trees, self._simple_trees, kinds=root_kinds)

        if self._print_trees:
            self._printer.prt('Dependency trees:\n', wrapped=True)
            self._printer.tree(self._simple_trees)

        unsorted_call_stack = OrderedDict()
        self._max_depth_all = -1
        for tag in self._model:
            model_tag = self._model[tag]
            roots = []
            if model_tag['kind'] in root_kinds:
                max_depth = 0
                roots = [model_tag['kind']]
            else:
                max_depth = -1
                for tag2 in self._trees:
                    if self.in_tree(tag, self._trees[tag2]):
                        roots.extend(self._trees[tag2]['roots'])
                    depth = self.get_max_depth(tag, self._trees[tag2],
                                               max_depth)
                    if depth > max_depth:
                        max_depth = depth
                    if depth > self._max_depth_all:
                        self._max_depth_all = depth
            roots = list(sorted(set(roots)))
            new_entry = deepcopy(model_tag)
            new_entry['roots'] = roots
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

        # with open(os.path.join(
        #         self.MODEL_OUTPUT_DIR,
        #         self._model_name + '-stack.json'), 'w') as f:
        #     json.dump(self._call_stack, f)

        for task in self._call_stack:
            cur_task = self._call_stack[task]
            mod_name = cur_task.get('class', task)
            if cur_task[
                    'kind'] == 'parameter' and task in self._parameter_json:
                cur_task.update(self._parameter_json[task])
            self._modules[task] = self._load_task_module(task)
            if mod_name == 'photometry':
                self._instruments = self._modules[task].instruments()
                self._bands = self._modules[task].bands()
            self._modules[task].set_attributes(cur_task)

            # This is currently not functional for MPI
            # cur_task = self._call_stack[task]
            # mod_name = cur_task.get('class', task)
            # mod_path = os.path.join('modules', cur_task['kind'] + 's',
            #                         mod_name + '.py')
            # if not os.path.isfile(mod_path):
            #     mod_path = os.path.join(self._dir_path, 'modules',
            #                             cur_task['kind'] + 's',
            #                             mod_name + '.py')
            # mod_name = ('mosfit.modules.' + cur_task['kind'] + 's.' +
            # mod_name)
            # mod = importlib.machinery.SourceFileLoader(mod_name,
            #                                            mod_path).load_module()
            # mod_class = getattr(mod, class_name)
            # if (cur_task['kind'] == 'parameter' and task in
            #         self._parameter_json):
            #     cur_task.update(self._parameter_json[task])
            # self._modules[task] = mod_class(name=task, **cur_task)
            # if mod_name == 'photometry':
            #     self._bands = self._modules[task].bands()

        # Look forward to see which modules want dense arrays.
        for task in self._call_stack:
            for ftask in self._call_stack:
                if (task != ftask and self._call_stack[ftask]['depth'] <
                        self._call_stack[task]['depth'] and
                        self._modules[ftask]._wants_dense):
                    self._modules[ftask]._provide_dense = True

    def _load_task_module(self, task, call_stack=None):
        if not call_stack:
            call_stack = self._call_stack
        cur_task = call_stack[task]
        kinds = self._inflect.plural(cur_task['kind'])
        mod_name = cur_task.get('class', task).lower()
        mod_path = os.path.join('modules', kinds, mod_name + '.py')
        if not os.path.isfile(mod_path):
            mod_path = os.path.join(self._dir_path, 'modules', kinds,
                                    mod_name + '.py')
        mod_name = 'mosfit.modules.' + kinds + mod_name
        try:
            mod = importlib.machinery.SourceFileLoader(mod_name,
                                                       mod_path).load_module()
        except AttributeError:
            import imp
            mod = imp.load_source(mod_name, mod_path)

        class_name = [
            x[0] for x in
            inspect.getmembers(mod, inspect.isclass)
            if issubclass(x[1], Module) and
            x[1].__module__ == mod.__name__][0]
        mod_class = getattr(mod, class_name)
        return mod_class(
            name=task, model=self, **cur_task)

    def create_data_dependent_free_parameters(
            self, variance_for_each=[], output={}):
        """Create free parameters that depend on loaded data."""
        unique_band_indices = list(
            sorted(set(output.get('all_band_indices', []))))
        needs_general_variance = any(
            np.array(output.get('all_band_indices', [])) < 0)

        new_call_stack = OrderedDict()
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if (cur_task.get('class', '') == 'variance' and
                    'band' in listify(variance_for_each)):
                # Find photometry in call stack.
                for ptask in self._call_stack:
                    if ptask == 'photometry':
                        awaves = self._modules[ptask].average_wavelengths(
                            unique_band_indices)
                        abands = self._modules[ptask].bands(
                            unique_band_indices)
                        band_pairs = list(sorted(zip(awaves, abands)))
                        break
                owav = 0.0
                variance_bands = []
                for bi, (awav, band) in enumerate(band_pairs):
                    wave_frac_diff = abs(awav - owav) / (awav + owav)
                    if wave_frac_diff < self.MIN_WAVE_FRAC_DIFF:
                        continue
                    new_task_name = '-'.join([task, 'band', band])
                    new_task = deepcopy(cur_task)
                    new_call_stack[new_task_name] = new_task
                    if 'latex' in new_task:
                        new_task['latex'] += '_{\\rm ' + band + '}'
                    new_call_stack[new_task_name] = new_task
                    self._modules[new_task_name] = self._load_task_module(
                        new_task_name, call_stack=new_call_stack)
                    owav = awav
                    variance_bands.append([awav, band])
                if needs_general_variance:
                    new_call_stack[task] = deepcopy(cur_task)
                if self._pool.is_master():
                    self._printer.message(
                        'anchoring_variances',
                        [', '.join([x[1] for x in variance_bands])],
                        wrapped=True)
                self._modules[ptask].set_variance_bands(variance_bands)
            else:
                new_call_stack[task] = deepcopy(cur_task)
        self._call_stack = new_call_stack

    def determine_number_of_measurements(self):
        """Estimate the number of measurements."""
        self._num_measurements = 0
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['kind'] == 'data':
                self._num_measurements += len(
                    self._modules[task]._data['times'])

    def determine_free_parameters(self, extra_fixed_parameters):
        """Generate `_free_parameters` and `_num_free_parameters`."""
        self._free_parameters = []
        self._num_variances = 0
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if (task not in extra_fixed_parameters and
                    cur_task['kind'] == 'parameter' and
                    'min_value' in cur_task and 'max_value' in cur_task and
                    cur_task['min_value'] != cur_task['max_value']):
                self._free_parameters.append(task)
                if cur_task.get('class', '') == 'variance':
                    self._num_variances += 1
        self._num_free_parameters = len(self._free_parameters)

    def exchange_requests(self):
        """Exchange requests between modules."""
        for task in reversed(self._call_stack):
            cur_task = self._call_stack[task]
            if 'requests' in cur_task:
                requests = OrderedDict()
                reqs = cur_task['requests']
                for req in reqs:
                    if reqs[req] not in self._modules:
                        raise RuntimeError(
                            'Request cannot be satisfied because module '
                            '`{}` could not be found.'.format(reqs[req]))
                    requests[req] = self._modules[reqs[req]].send_request(req)
                self._modules[task].receive_requests(**requests)

    def frack(self, arg):
        """Perform fracking upon a single walker.

        Uses a randomly-selected global or local minimization method.
        """
        x = np.array(arg[0])
        step = 1.0
        seed = arg[1]
        np.random.seed(seed)
        my_choice = np.random.choice(range(3))
        # my_choice = 0
        my_method = ['L-BFGS-B', 'TNC', 'SLSQP'][my_choice]
        opt_dict = {'disp': False, 'approx_grad': True}
        if my_method in ['TNC', 'SLSQP']:
            opt_dict['maxiter'] = 200
        elif my_method == 'L-BFGS-B':
            opt_dict['maxfun'] = 5000
            opt_dict['maxls'] = 50
        # bounds = [(0.0, 1.0) for y in range(self._num_free_parameters)]
        bounds = list(
            zip(np.clip(x - step, 0.0, 1.0), np.clip(x + step, 0.0, 1.0)))

        bh = minimize(
            self.fprob,
            x,
            method=my_method,
            bounds=bounds,
            options=opt_dict)

        # bounds = list(
        #     zip(np.clip(x - step, 0.0, 1.0), np.clip(x + step, 0.0, 1.0)))
        #
        # bh = differential_evolution(
        #     self.fprob, bounds, disp=True, polish=False)

        # bh = basinhopping(
        #     self.fprob,
        #     x,
        #     disp=True,
        #     niter=10,
        #     minimizer_kwargs={'method': "L-BFGS-B",
        #                       'bounds': bounds})

        # bo = BayesianOptimization(self.boprob, dict(
        #     [('x' + str(i),
        #       (np.clip(x[i] - step, 0.0, 1.0),
        #        np.clip(x[i] + step, 0.0, 1.0)))
        #      for i in range(len(x))]))
        #
        # bo.explore(dict([('x' + str(i), [x[i]]) for i in range(len(x))]))
        #
        # bo.maximize(init_points=0, n_iter=20, acq='ei')
        #
        # bh = self.outClass()
        # bh.x = [x[1] for x in sorted(bo.res['max']['max_params'].items())]
        # bh.fun = -bo.res['max']['max_val']

        # m = Minuit(self.fprob)
        # m.migrad()
        return bh

    def construct_trees(
            self, d, trees, simple, kinds=[], name='', roots=[], depth=0):
        """Construct call trees for each root."""
        leaf = kinds if len(kinds) else name
        if depth > 100:
            raise RuntimeError(
                'Error: Tree depth greater than 100, suggests a recursive '
                'input loop in `{}`.'.format(leaf))
        for tag in d:
            entry = deepcopy(d[tag])
            new_roots = list(roots)
            if entry['kind'] in kinds or tag == name:
                entry['depth'] = depth
                if entry['kind'] in kinds:
                    new_roots.append(entry['kind'])
                entry['roots'] = list(sorted(set(new_roots)))
                trees[tag] = entry
                simple[tag] = OrderedDict()
                inputs = listify(entry.get('inputs', []))
                for inp in inputs:
                    if inp not in d:
                        suggests = get_close_matches(inp, d, n=1, cutoff=0.8)
                        warn_str = (
                            'Module `{}` for input to `{}` '
                            'not found!'.format(inp, leaf))
                        if len(suggests):
                            warn_str += (
                                ' Did you perhaps mean `{}`?'.
                                format(suggests[0]))
                        raise RuntimeError(warn_str)
                    children = OrderedDict()
                    simple_children = OrderedDict()
                    self.construct_trees(
                        d,
                        children,
                        simple_children,
                        name=inp,
                        roots=new_roots,
                        depth=depth + 1)
                    trees[tag].setdefault('children', OrderedDict())
                    trees[tag]['children'].update(children)
                    simple[tag].update(simple_children)

    def draw_walker(self, test=True, walkers_pool=[], replace=False):
        """Draw a walker randomly.

        Draw a walker randomly from the full range of all parameters, reject
        walkers that return invalid scores.
        """
        p = None
        chosen_one = None
        while p is None:
            draw = np.random.uniform(
                low=0.0, high=1.0, size=self._num_free_parameters)
            draw = [
                self._modules[self._free_parameters[i]].prior_cdf(x)
                for i, x in enumerate(draw)
            ]
            if len(walkers_pool):
                chosen_one = np.random.choice(range(len(walkers_pool)))
                for e, elem in enumerate(walkers_pool[chosen_one]):
                    if elem is not None:
                        draw[e] = elem
            if not test:
                p = draw
                score = None
                break
            score = self.likelihood(draw)
            if (not isnan(score) and np.isfinite(score) and
                (not isinstance(self._fitter._draw_above_likelihood, float) or
                 score > self._fitter._draw_above_likelihood)):
                p = draw

        if not replace and chosen_one is not None:
            del(walkers_pool[chosen_one])
        return (p, score)

    def get_max_depth(self, tag, parent, max_depth):
        """Return the maximum depth a given task is found in a tree."""
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

    def in_tree(self, tag, parent):
        """Return the maximum depth a given task is found in a tree."""
        for child in parent.get('children', []):
            if child == tag:
                return True
            else:
                if self.in_tree(tag, parent['children'][child]):
                    return True
        return False

    def pool(self):
        """Return processing pool."""
        return self._pool

    def printer(self):
        """Return printer."""
        return self._printer

    def likelihood(self, x):
        """Return score related to maximum likelihood."""
        outputs = self.run_stack(x, root='objective')
        return outputs['value']

    def prior(self, x):
        """Return score related to paramater priors."""
        prior = 0.0
        for pi, par in enumerate(self._free_parameters):
            lprior = self._modules[par].lnprior_pdf(x[pi])
            prior = prior + lprior
        return prior

    def boprob(self, **kwargs):
        """Score for `BayesianOptimization`."""
        x = []
        for key in sorted(kwargs):
            x.append(kwargs[key])

        l = self.likelihood(x) + self.prior(x)
        if not np.isfinite(l):
            return LOCAL_LIKELIHOOD_FLOOR
        return l

    def fprob(self, x):
        """Return score for fracking."""
        l = -(self.likelihood(x) + self.prior(x))
        if not np.isfinite(l):
            return -LOCAL_LIKELIHOOD_FLOOR
        return l

    def plural(self, x):
        """Pluralize and cache model-related keys."""
        if x not in self._inflections:
            plural = self._inflect.plural(x)
            if plural == x:
                plural = x + 's'
            self._inflections[x] = plural
        else:
            plural = self._inflections[x]
        return plural

    def run_stack(self, x, root='objective'):
        """Run module stack.

        Run a stack of modules as defined in the model definition file. Only
        run functions that match the specified root.
        """
        inputs = OrderedDict()
        outputs = OrderedDict()
        pos = 0
        cur_depth = self._max_depth_all
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if root not in cur_task['roots']:
                continue
            if cur_task['depth'] != cur_depth:
                inputs = outputs
            inputs.update(OrderedDict([('root', root)]))
            cur_depth = cur_task['depth']
            if task in self._free_parameters:
                inputs.update(OrderedDict([('fraction', x[pos])]))
                inputs.setdefault('fractions', []).append(x[pos])
                pos = pos + 1
            try:
                new_outs = self._modules[task].process(**inputs)
                if not isinstance(new_outs, OrderedDict):
                    new_outs = OrderedDict(sorted(new_outs.items()))
            except Exception:
                self._printer.prt(
                    "Failed to execute module `{}`\'s process().".format(task),
                    wrapped=True)
                raise

            outputs.update(new_outs)

            # Append module references
            self._references.extend(self._modules[task]._REFERENCES)

            if '_delete_keys' in outputs:
                for key in outputs['_delete_keys']:
                    del(outputs[key])
                del(outputs['_delete_keys'])

            if cur_task['kind'] == root:
                # Make sure references are unique.
                self._references = list(map(
                    dict, set(tuple(sorted(d.items()))
                              for d in self._references)))
                return outputs
