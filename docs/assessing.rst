.. _assessing:

================
Assessing models
================

.. _convergence:

-----------
Convergence
-----------

The meaning of ``-R`` / ``--run-until-converged`` depends on the sampler
(see :ref:`sampling`).

For the **default** nested sampler (``-D dynesty``), ``-R`` is an evidence
threshold :math:`\Delta\log Z` (``dlogz``). Passing ``-R`` with no value uses
``0.02``. That threshold is used both for the static (baselining) phase and
as the evidence-error target in the dynamic (batching) phase. Dynesty does
not use Gelman–Rubin / PSRF. Nested sampling also does not use ``-i`` as a
hard iteration cap in the same way ensemble MCMC does; ``-R`` is the usual
way to run until the remaining evidence is small. ``--num-walkers`` / ``-N``
does not set the number of live points (``nlive = 20 \times`` the number of
free parameters).

For **ensemble MCMC** (``-D ensembler``), convergence is assessed using the
Gelman–Rubin statistic (or "potential scale reduction factor", abbreviated
PSRF), which is a measure of the in-chain variance as compared to the
between-chain variance. This metric is calculated for each free parameter,
with the global PSRF score being derived by taking the maximum difference
amongst all the individual parameter PSRFs. If a model is converged and
well-mixed, these two values should be close to equal (PSRF ~ 1), and any
significant deviance from equality suggests that the chains have yet to
converge.

Without ``-R``, an ensembler run uses a fixed number of iterations
(``-i``; default a few thousand) so the total runtime is deterministic. If
``-R`` is passed with ``-D ensembler``, the code continues beyond that
iteration limit until the PSRF is less than a prescribed value (by default
1.1, unless the user sets another value using ``-R``).

Another ensembler measure of convergence is the autocorrelation time
:math:`\tau_{\rm auto}`, estimated using the ``acor`` function embedded
within ``emcee``. Unfortunately, this metric usually does not give an
indication of how close one is to convergence until one is already
converged, as it fails to yield an estimate for the autocorrelation time if
:math:`\tau_{\rm auto} > i`, where :math:`i` is the number of iterations.
We find that typically chains must run for significantly longer than what is
required to converge according to the PSRF before ``acor`` will yield a
numerical value (``-R 1.05`` or less).

The fact that ``acor`` does not yield a value until the PSRF ~ 1 means that
the number of independent draws from the posterior is significantly
constrained unless the user chooses to run their chains for much longer.
With ``-D ensembler``, ``MOSFiT`` can be instructed to run until a certain
number of independent samples are available via the ``-U`` option.
``-R`` and ``-U`` cannot be combined.

.. _scoring:

-------
Scoring
-------

Model compatibility with a given dataset is measured using the "Watanabe-Akaike information criterion" (WAIC, also known as the "widely applicable information criterion", [WAT2010]_), which is simply the score of the parameter combination with the highest likelihood minus the variance of the scores within the fully-converged posterior. Ideally, one prefers models with the fewest free parameters, the WAIC estimates the *effective* number of free parameters for a given model and adjusts the score accordingly. In principle, two models with the same score for their best fits may have wildly different WAIC scores depending on the distribution of scores within their posteriors. This criterion is less sensitive to overfitting than simply comparing the best scores yielded by two models, and should also provide a fair comparison between models with different numbers of free parameters.

.. [WAT2010] `Watanabe et al. 2010 <http://www.jmlr.org/papers/v11/watanabe10a.html>`_
