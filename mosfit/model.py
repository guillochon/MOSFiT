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
from astrocats.catalog.quantity import QUANTITY
from mosfit.constants import LOCAL_LIKELIHOOD_FLOOR
from mosfit.modules.module import Module
from mosfit.utils import is_number, listify, pretty_num
from schwimmbad import SerialPool
# from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from six import string_types


# from scipy.optimize import basinhopping

# from bayes_opt import BayesianOptimization


class Model(object):
    """Define a semi-analytical model to fit transients with."""

    MODEL_OUTPUT_DIR = 'products'
    MIN_WAVE_FRAC_DIFF = 0.1
    DRAW_LIMIT = 10

    # class outClass(object):
    #     pass

    def __init__(self,
                 parameter_path='parameters.json',
                 model='',
                 data={},
                 wrap_length=100,
                 pool=None,
                 test=False,
                 printer=None,
                 fitter=None,
                 print_trees=False):
        """Initialize `Model` object."""
        from mosfit.fitter import Fitter

        self._model_name = model
        self._pool = SerialPool() if pool is None else pool
        self._is_master = pool.is_master() if pool else False
        self._wrap_length = wrap_length
        self._print_trees = print_trees
        self._inflect = inflect.engine()
        self._test = test
        self._inflections = {}
        self._references = OrderedDict()
        self._free_parameters = []
        self._user_fixed_parameters = []
        self._kinds_needed = set()
        self._kinds_supported = set()

        self._draw_limit_reached = False

        self._fitter = Fitter() if not fitter else fitter
        self._printer = self._fitter._printer if not printer else printer

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

        # Create list of all available models.
        all_models = set()
        if os.path.isdir('models'):
            all_models |= set(next(os.walk('models'))[1])
        models_path = os.path.join(self._dir_path, 'models')
        if os.path.isdir(models_path):
            all_models |= set(next(os.walk(models_path))[1])
        all_models = list(sorted(list(all_models)))

        if not self._model_name:
            claimed_type = None
            try:
                claimed_type = list(data.values())[0][
                    'claimedtype'][0][QUANTITY.VALUE]
            except Exception:
                prt.message('no_model_type', warning=True)

            all_models_txt = prt.text('all_models')
            suggested_models_txt = prt.text('suggested_models', [claimed_type])
            another_model_txt = prt.text('another_model')

            type_options = model_types.get(
                claimed_type, []) if claimed_type else []
            if not type_options:
                type_options = all_models
                model_prompt_txt = all_models_txt
            else:
                type_options.append(another_model_txt)
                model_prompt_txt = suggested_models_txt
            if not type_options:
                prt.message('no_model_for_type', warning=True)
            else:
                while not self._model_name:
                    if self._test:
                        self._model_name = type_options[0]
                    else:
                        sel = self._printer.prompt(
                            model_prompt_txt,
                            kind='option',
                            options=type_options,
                            message=False,
                            default='n',
                            none_string=('None of the above, skip this '
                                         'transient.'))
                        if sel is not None:
                            self._model_name = type_options[sel - 1]
                    if not self._model_name:
                        break
                    if self._model_name == another_model_txt:
                        type_options = all_models
                        model_prompt_txt = all_models_txt
                        self._model_name = None

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

        # Find @ tags, store them, and prune them from `_model`.
        for tag in list(self._model.keys()):
            if tag.startswith('@'):
                if tag == '@references':
                    self._references.setdefault('base', []).extend(
                        self._model[tag])
                del(self._model[tag])

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
        self._telescopes = []

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
                self._telescopes = self._modules[task].telescopes()
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

        # Count free parameters.
        self.determine_free_parameters()

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
            name=task, model=self, fitter=self._fitter, **cur_task)

    def load_data(self,
                  data,
                  event_name='',
                  smooth_times=-1,
                  extrapolate_time=0.0,
                  limit_fitting_mjds=False,
                  exclude_bands=[],
                  exclude_instruments=[],
                  exclude_systems=[],
                  exclude_sources=[],
                  exclude_kinds=[],
                  band_list=[],
                  band_systems=[],
                  band_instruments=[],
                  band_bandsets=[],
                  band_sampling_points=17,
                  variance_for_each=[],
                  user_fixed_parameters=[],
                  pool=None):
        """Load the data for the specified event."""
        if pool is not None:
            self._pool = pool
            self._printer._pool = pool

        prt = self._printer

        prt.message('loading_data', inline=True)

        # Fix user-specified parameters.
        fixed_parameters = []
        for task in self._call_stack:
            for fi, param in enumerate(user_fixed_parameters):
                if (task == param or
                        self._call_stack[task].get(
                            'class', '') == param):
                    fixed_parameters.append(task)
                    if fi < len(user_fixed_parameters) - 1 and is_number(
                            user_fixed_parameters[fi + 1]):
                        value = float(user_fixed_parameters[fi + 1])
                        if value not in self._call_stack:
                            self._call_stack[task]['value'] = value
                    if 'min_value' in self._call_stack[task]:
                        del self._call_stack[task]['min_value']
                    if 'max_value' in self._call_stack[task]:
                        del self._call_stack[task]['max_value']
                    self._modules[task].fix_value(
                        self._call_stack[task]['value'])

        self.determine_free_parameters(fixed_parameters)

        for ti, task in enumerate(self._call_stack):
            cur_task = self._call_stack[task]
            self._modules[task].set_event_name(event_name)
            new_per = np.round(100.0 * float(ti) / len(self._call_stack))
            prt.message('loading_task', [task, new_per], inline=True)
            self._kinds_supported |= set(cur_task.get('supports', []))
            if cur_task['kind'] == 'data':
                success = self._modules[task].set_data(
                    data,
                    req_key_values=OrderedDict((
                        ('band', self._bands),
                        ('instrument', self._instruments),
                        ('telescope', self._telescopes))),
                    subtract_minimum_keys=['times'],
                    smooth_times=smooth_times,
                    extrapolate_time=extrapolate_time,
                    limit_fitting_mjds=limit_fitting_mjds,
                    exclude_bands=exclude_bands,
                    exclude_instruments=exclude_instruments,
                    exclude_systems=exclude_systems,
                    exclude_sources=exclude_sources,
                    exclude_kinds=exclude_kinds,
                    band_list=band_list,
                    band_systems=band_systems,
                    band_instruments=band_instruments,
                    band_bandsets=band_bandsets)
                if not success:
                    return False
                fixed_parameters.extend(self._modules[task]
                                        .get_data_determined_parameters())
            elif cur_task['kind'] == 'sed':
                self._modules[task].set_data(band_sampling_points)
            self._kinds_needed |= self._modules[task]._kinds_needed

        # Find unsupported wavebands and report to user.
        unsupported_kinds = self._kinds_needed - self._kinds_supported
        if len(unsupported_kinds):
            prt.message(
                'using_unsupported_kinds' if 'none' in exclude_kinds else
                'ignoring_unsupported_kinds', [', '.join(
                    sorted(unsupported_kinds))], warning=True)

        # Determine free parameters again as setting data may have fixed some
        # more.
        self.determine_free_parameters(fixed_parameters)

        self.exchange_requests()

        prt.message('finding_bands', inline=True)

        # Run through once to set all inits.
        for root in ['output', 'objective']:
            outputs = self.run_stack(
                [0.0 for x in range(self._num_free_parameters)],
                root=root)

        # Create any data-dependent free parameters.
        self.adjust_fixed_parameters(variance_for_each, outputs)

        # Determine free parameters again as above may have changed them.
        self.determine_free_parameters(fixed_parameters)

        self.determine_number_of_measurements()

        self.exchange_requests()

        # Reset modules
        for task in self._call_stack:
            self._modules[task].reset_preprocessed(['photometry'])

        # Run through inits once more.
        for root in ['output', 'objective']:
            outputs = self.run_stack(
                [0.0 for x in range(self._num_free_parameters)],
                root=root)

        # Collect observed band info
        if self._pool.is_master() and 'photometry' in self._modules:
            prt.message('bands_used')
            bis = list(
                filter(lambda a: a != -1,
                       sorted(set(outputs['all_band_indices']))))
            ois = []
            for bi in bis:
                ois.append(
                    any([
                        y
                        for x, y in zip(outputs['all_band_indices'], outputs[
                            'observed']) if x == bi
                    ]))
            band_len = max([
                len(self._modules['photometry']._unique_bands[bi][
                    'origin']) for bi in bis
            ])
            filts = self._modules['photometry']
            ubs = filts._unique_bands
            filterarr = [(ubs[bis[i]]['systems'], ubs[bis[i]]['bandsets'],
                          filts._average_wavelengths[bis[i]],
                          filts._band_offsets[bis[i]],
                          filts._band_kinds[bis[i]],
                          filts._band_names[bis[i]],
                          ois[i], bis[i])
                         for i in range(len(bis))]
            filterrows = [(
                ' ' + (' ' if s[-2] else '*') + ubs[s[-1]]['origin']
                .ljust(band_len) + ' [' + ', '.join(
                    list(
                        filter(None, (
                            'Bandset: ' + s[1] if s[1] else '',
                            'System: ' + s[0] if s[0] else '',
                            'AB offset: ' + pretty_num(
                                s[3]) if (s[4] == 'magnitude' and
                                          s[0] != 'AB') else '')))) +
                ']').replace(' []', '') for s in list(sorted(filterarr))]
            if not all(ois):
                filterrows.append(prt.text('not_observed'))
            prt.prt('\n'.join(filterrows))

            single_freq_inst = list(
                sorted(set(np.array(outputs['instruments'])[
                    np.array(outputs['all_band_indices']) == -1])))

            if len(single_freq_inst):
                prt.message('single_freq')
            for inst in single_freq_inst:
                prt.prt('  {}'.format(inst))

            if ('unmatched_bands' in outputs and
                    'unmatched_instruments' in outputs):
                prt.message('unmatched_obs', warning=True)
                prt.prt(', '.join(
                    ['{} [{}]'.format(x[0], x[1]) if x[0] and x[1] else x[0]
                     if not x[1] else x[1] for x in list(set(zip(
                         outputs['unmatched_bands'],
                         outputs['unmatched_instruments'])))]), warning=True,
                    prefix=False, wrapped=True)

        return True

    def adjust_fixed_parameters(
            self, variance_for_each=[], output={}):
        """Create free parameters that depend on loaded data."""
        unique_band_indices = list(
            sorted(set(output.get('all_band_indices', []))))
        needs_general_variance = any(
            np.array(output.get('all_band_indices', [])) < 0)

        new_call_stack = OrderedDict()
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            vfe = listify(variance_for_each)
            if task == 'variance' and 'band' in vfe:
                vfi = vfe.index('band') + 1
                mwfd = float(vfe[vfi]) if (vfi < len(vfe) and is_number(
                    vfe[vfi])) else self.MIN_WAVE_FRAC_DIFF
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
                    if wave_frac_diff < mwfd:
                        continue
                    new_task_name = '-'.join([task, 'band', band])
                    if new_task_name in self._call_stack:
                        continue
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
            # Fixed any variables to be fixed if any conditional inputs are
            # fixed by the data.
            # if any([listify(x)[-1] == 'conditional'
            #         for x in cur_task.get('inputs', [])]):
        self._call_stack = new_call_stack

        for task in reversed(self._call_stack):
            cur_task = self._call_stack[task]
            for inp in cur_task.get('inputs', []):
                other = listify(inp)[0]
                if (cur_task['kind'] == 'parameter' and
                        output.get(other, None) is not None):
                    if (not self._modules[other]._fixed or
                            self._modules[other]._fixed_by_user):
                        self._modules[task]._fixed = True
                    self._modules[task]._derived_keys = list(set(
                        self._modules[task]._derived_keys + [task]))

    def determine_number_of_measurements(self):
        """Estimate the number of measurements."""
        self._num_measurements = 0
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if cur_task['kind'] == 'data':
                self._num_measurements += len(
                    self._modules[task]._data['times'])

    def determine_free_parameters(self, extra_fixed_parameters=[]):
        """Generate `_free_parameters` and `_num_free_parameters`."""
        self._free_parameters = []
        self._user_fixed_parameters = []
        self._num_variances = 0
        for task in self._call_stack:
            cur_task = self._call_stack[task]
            if (task not in extra_fixed_parameters and
                    cur_task['kind'] == 'parameter' and
                    'min_value' in cur_task and 'max_value' in cur_task and
                    cur_task['min_value'] != cur_task['max_value'] and
                    not self._modules[task]._fixed):
                self._free_parameters.append(task)
                if cur_task.get('class', '') == 'variance':
                    self._num_variances += 1
            elif (cur_task['kind'] == 'parameter' and
                  task in extra_fixed_parameters):
                self._user_fixed_parameters.append(task)
        self._num_free_parameters = len(self._free_parameters)

    def is_parameter_fixed_by_user(self, parameter):
        """Return whether a parameter is fixed by the user."""
        return parameter in self._user_fixed_parameters

    def get_num_free_parameters(self):
        """Return number of free parameters."""
        return self._num_free_parameters

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
                for inps in inputs:
                    conditional = False
                    if isinstance(inps, list) and not isinstance(
                            inps, string_types) and inps[-1] == "conditional":
                        inp = inps[0]
                        conditional = True
                    else:
                        inp = inps
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
                    # Conditional inputs don't propagate down the tree.
                    if conditional:
                        continue
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
        draw_cnt = 0
        while p is None:
            draw_cnt += 1
            draw = np.random.uniform(
                low=0.0, high=1.0, size=self._num_free_parameters)
            draw = [
                self._modules[self._free_parameters[i]].prior_cdf(x)
                for i, x in enumerate(draw)
            ]
            if len(walkers_pool):
                if not replace:
                    chosen_one = 0
                else:
                    chosen_one = np.random.choice(range(len(walkers_pool)))
                for e, elem in enumerate(walkers_pool[chosen_one]):
                    if elem is not None:
                        draw[e] = elem
            if not test:
                p = draw
                score = None
                break
            score = self.ln_likelihood(draw)
            if draw_cnt >= self.DRAW_LIMIT and not self._draw_limit_reached:
                self._printer.message('draw_limit_reached', warning=True)
                self._draw_limit_reached = True
            if ((not isnan(score) and np.isfinite(score) and
                 (not isinstance(self._fitter._draw_above_likelihood, float) or
                  score > self._fitter._draw_above_likelihood)) or
                    draw_cnt >= self.DRAW_LIMIT):
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

    def run(self, x, root='output'):
        """Run stack with the given root."""
        outputs = self.run_stack(x, root=root)
        return outputs

    def printer(self):
        """Return printer."""
        return self._printer

    def likelihood(self, x):
        """Return score related to maximum likelihood."""
        return np.exp(self.ln_likelihood(x))

    def ln_likelihood(self, x):
        """Return ln(likelihood)."""
        outputs = self.run_stack(x, root='objective')
        return outputs['value']

    def free_parameter_names(self, x):
        """Return list of free parameter names."""
        return self._free_parameters

    def prior(self, x):
        """Return score related to paramater priors."""
        return np.exp(self.ln_prior(x))

    def ln_prior(self, x):
        """Return ln(prior)."""
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

        l = self.ln_likelihood(x) + self.ln_prior(x)
        if not np.isfinite(l):
            return LOCAL_LIKELIHOOD_FLOOR
        return l

    def fprob(self, x):
        """Return score for fracking."""
        l = -(self.ln_likelihood(x) + self.ln_prior(x))
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

    def reset_unset_recommended_keys(self):
        """Null the list of unset recommended keys across all modules."""
        for module in self._modules.values():
            module.reset_unset_recommended_keys()

    def get_unset_recommended_keys(self):
        """Collect list of unset recommended keys across all modules."""
        unset_keys = set()
        for module in self._modules.values():
            unset_keys.update(module.get_unset_recommended_keys())
        return unset_keys

    def run_stack(self, x, root='objective'):
        """Run module stack.

        Run a stack of modules as defined in the model definition file. Only
        run functions that match the specified root.
        """
        inputs = OrderedDict()
        outputs = OrderedDict()
        pos = 0
        cur_depth = self._max_depth_all

        # If this is the first time running this stack, build the ref arrays.
        build_refs = root not in self._references
        if build_refs:
            self._references[root] = []

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
            if build_refs:
                self._references[root].extend(self._modules[task]._REFERENCES)

            if '_delete_keys' in outputs:
                for key in list(outputs['_delete_keys'].keys()):
                    del(outputs[key])
                del(outputs['_delete_keys'])

        if build_refs:
            # Make sure references are unique.
            self._references[root] = list(map(dict, set(tuple(
                sorted(d.items())) for d in self._references[root])))

        return outputs
