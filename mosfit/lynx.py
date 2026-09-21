"""Rest-frame SED interface for external light-curve simulators.

MOSFiT's own pipeline takes a model all the way to observed photometry: it
redshifts the SED, applies line-of-sight extinction, integrates through
bandpasses and compares against data. Simulators such as `LightCurveLynx
<https://lightcurvelynx.readthedocs.io>`_ instead want a *source*, and do all
of that themselves. Their ``SEDModel`` contract is a single call,

    compute_sed(times, wavelengths, graph_state) -> (T, N) array of nJy

evaluated in the rest frame, with no dust and no distance dimming beyond the
10 pc absolute-flux convention.

This module provides that view without disturbing the default behaviour. The
same machinery backs the ``--lynx`` command-line flag; :class:`LynxSource` is
the in-process entry point a wrapper should build on, since it avoids the
per-call file output and printer traffic of a full MOSFiT run.
"""
from collections import OrderedDict

import numpy as np

# Parameters that the calling simulator is responsible for, pinned so they
# cannot vary underneath it. ``lumdist`` is 10 pc in Mpc, which the
# ``Redshift`` module already treats as the zero-redshift limit.
LYNX_FIXED_PARAMETERS = OrderedDict([
    ('redshift', 0.0),
    ('lumdist', 1.0e-5),
    ('texplosion', 0.0),
    ('ebv', 0.0),
    ('rvhost', 3.1),
    ('nhhost', 1.0e16),
    ('variance', 1.0e-3),
])

#: ``(min, max, n)`` in Angstroms; spans the Rubin ``u`` through ``y`` range
#: with room on either side for redshifting by the caller.
DEFAULT_WAVELENGTH_GRID = (1000.0, 25000.0, 100)


def lynx_wavelength_grid(spec=None):
    """Build the rest-frame wavelength grid from a ``(min, max, n)`` spec."""
    if spec is None:
        spec = DEFAULT_WAVELENGTH_GRID
    spec = list(spec)
    if len(spec) != 3:
        raise ValueError(
            '`--lynx-wavelengths` takes exactly three values: MIN MAX N.')
    min_wave, max_wave, n_wave = float(spec[0]), float(spec[1]), int(spec[2])
    if min_wave <= 0.0 or max_wave <= min_wave:
        raise ValueError(
            'Lynx wavelength range must satisfy 0 < MIN < MAX (got '
            '{} and {}).'.format(min_wave, max_wave))
    if n_wave < 2:
        raise ValueError(
            'Lynx wavelength grid needs at least two samples (got '
            '{}).'.format(n_wave))
    return np.linspace(min_wave, max_wave, n_wave)


class LynxSource(object):
    """A MOSFiT model exposed as a rest-frame SED source.

    Parameters
    ----------
    model : str
        Name of the MOSFiT model, e.g. ``'slsn'``.
    wavelengths : array_like, optional
        Rest-frame wavelength grid in Angstroms. Defaults to
        :data:`DEFAULT_WAVELENGTH_GRID`.
    phases : array_like, optional
        Phases in days since explosion at which the model is set up. Calls to
        :meth:`compute_sed` with a different grid rebuild it, which is
        expensive, so pass the grid you intend to use.
    parameter_path : str
        Prior file to read parameter ranges from.

    Notes
    -----
    Rebuilding the time grid re-runs MOSFiT's data-loading path, so a wrapper
    should hold one instance per grid and vary only the parameters.
    """

    def __init__(self, model='slsn', wavelengths=None, phases=None,
                 parameter_path='parameters.json', quiet=True,
                 max_time=200.0, n_phases=100, **kwargs):
        """Initialize `LynxSource`."""
        from mosfit.fitter import Fitter

        self._model_name = model
        self._parameter_path = parameter_path
        self._wavelengths = (
            lynx_wavelength_grid() if wavelengths is None
            else np.asarray(wavelengths, dtype=float))
        if phases is None:
            phases = np.linspace(0.0, float(max_time), int(n_phases))
        self._phases = np.asarray(phases, dtype=float)

        self._fitter = Fitter(
            quiet=quiet, lynx=True, lynx_wavelengths=self._wavelengths,
            **kwargs)
        self._model = None
        self._build()

    def _build(self):
        """Construct the underlying `Model` on the current grids."""
        from mosfit.model import Model

        self._fitter._lynx_wavelengths = self._wavelengths
        data = self._fitter.generate_dummy_data(
            self._model_name, max_time=float(np.max(self._phases)),
            time_list=list(self._phases))
        model = Model(
            model=self._model_name,
            data=data,
            parameter_path=self._parameter_path,
            fitter=self._fitter,
            printer=self._fitter._printer)
        fixed = []
        for name, value in LYNX_FIXED_PARAMETERS.items():
            fixed += [name, value]
        model.load_data(
            data, event_name=self._model_name, time_list=list(self._phases),
            user_fixed_parameters=fixed)
        self._model = model

    def _ensure_grid(self, times=None, wavelengths=None):
        """Rebuild if the caller asked for grids we were not set up on."""
        rebuild = False
        if wavelengths is not None:
            wavelengths = np.asarray(wavelengths, dtype=float)
            if not np.array_equal(wavelengths, self._wavelengths):
                self._wavelengths = wavelengths
                rebuild = True
        if times is not None:
            times = np.asarray(times, dtype=float)
            if not np.array_equal(times, self._phases):
                self._phases = times
                rebuild = True
        if rebuild:
            self._build()

    def parameter_manifest(self, include_fixed=True):
        """Return this model's parameters, ranges and unit-cube ordering."""
        return self._model.parameter_manifest(include_fixed=include_fixed)

    def free_parameter_names(self):
        """Return the free parameter names, in walker-vector order."""
        return list(self._model.free_parameter_names())

    def minwave(self):
        """Minimum sampled wavelength (Angstroms)."""
        return float(np.min(self._wavelengths))

    def maxwave(self):
        """Maximum sampled wavelength (Angstroms)."""
        return float(np.max(self._wavelengths))

    def minphase(self):
        """Earliest phase the model is set up for (days since explosion)."""
        return self._model.minphase()

    def maxphase(self):
        """Latest phase the model is set up for (days since explosion)."""
        return self._model.maxphase()

    def fractions_from_parameters(self, parameters=None):
        """Map physical parameter values onto MOSFiT's unit-cube vector.

        Unspecified free parameters take the midpoint of their prior, so a
        caller can vary a subset without having to name the rest.
        """
        parameters = dict(parameters or {})
        free_names = list(self._model.free_parameter_names())
        unknown = set(parameters) - set(free_names)
        if unknown:
            raise ValueError(
                'Not free parameters of model `{}`: {}. Free parameters are: '
                '{}.'.format(
                    self._model_name, ', '.join(sorted(unknown)),
                    ', '.join(free_names)))
        vector = []
        for name in free_names:
            module = self._model._modules[name]
            if name in parameters:
                vector.append(float(module.fraction(float(parameters[name]))))
            else:
                vector.append(0.5)
        return np.array(vector, dtype=float)

    def compute_sed(self, times=None, wavelengths=None, parameters=None,
                    fractions=None):
        """Return the rest-frame SED as a ``(n_time, n_wave)`` array in nJy.

        Parameters
        ----------
        times : array_like, optional
            Phases in days since explosion. Defaults to the construction grid.
        wavelengths : array_like, optional
            Rest-frame wavelengths in Angstroms.
        parameters : dict, optional
            Physical parameter values, keyed by MOSFiT parameter name.
        fractions : array_like, optional
            Unit-cube coordinates, as an alternative to ``parameters``. Takes
            precedence when both are given.
        """
        self._ensure_grid(times, wavelengths)
        if fractions is None:
            fractions = self.fractions_from_parameters(parameters)
        fractions = np.asarray(fractions, dtype=float)

        output = self._model.run_stack(fractions, root='output')
        seds = output.get('lynx_seds')
        if seds is None:
            raise RuntimeError(
                'Model produced no rest-frame SED; the `lynxsed` output task '
                'is missing from this model definition.')
        phases = np.asarray(output['lynx_phases'], dtype=float)
        seds = np.asarray(seds, dtype=float)

        # ``run_stack`` returns rows for the grid the model was built on, which
        # can carry endpoints the caller did not ask for, so realign explicitly
        # rather than assuming a one-to-one correspondence. Rows the model did
        # not produce stay zero, matching how pre-explosion phases behave.
        if phases.shape[0] != self._phases.shape[0]:
            out = np.zeros(
                (self._phases.shape[0], seds.shape[1]), dtype=float)
            if phases.shape[0]:
                idx = np.clip(
                    np.searchsorted(phases, self._phases),
                    0, phases.shape[0] - 1)
                hit = np.isclose(phases[idx], self._phases)
                out[hit] = seds[idx[hit]]
            return out
        return seds
