.. _assessing:

================
Assessing models
================

.. _convergence:

-----------
Convergence
-----------

Convergence in ``MOSFiT`` is assessed using the Gelman-Rubin statistic (or "potential scale reduction factor", abbreviated PSRF), which is a measure of the in-chain variance as compared to the between-chain variance. This metric is calculated for each free parameter, with the global PSRF score being derived by taking the maximum difference amongst all the individual parameter PSRFs. If a model is converged and well-mixed, these two values should be close to equal (PSRF ~ 1), and any significant deviance from equality suggests that the chains have yet to converge.

By default, ``MOSFiT`` will run for a small, fixed number of iterations (``-i 5000``), regardless of how well-converged a model is, to guarantee the total runtime is deterministic. If however the ``-R`` option is passed to ``MOSFiT``, the code will continue to evolve the chains beyond the iteration limit specified by ``-i`` until the PSRF is less than a prescribed value (by default 1.1, unless the user sets another value using ``-R``).

Another measure of convergence is the autocorrelation time :math:`\tau_{\rm auto}`, estimated using the ``acor`` function embedded within ``emcee``. Unfortunately, this metric usually does not give an indication of how close one is to convergence until one is already converged, as it fails to yield an estimate for the autocorrelation time if :math:`\tau_{\rm auto} > i`, where :math:`i` is the number of iterations. We find that typically chains must run for significantly longer than what is required to converge according to the PSRF before ``acor`` will yield a numerical value (``-R 1.05`` or less).

The fact that ``acor`` does not yield a value until the PSRF ~ 1 means that the number of independent draws from the posterior is significantly constrained unless the user chooses to run their chains for much longer. ``MOSFiT`` can be instructed to run until a certain number of independent samples are available via the ``-U`` option.

.. _scoring:

-------
Scoring
-------

Model compatibility with a given dataset is measured using the "Watanabe-Akaike information criterion" (WAIC, also known as the "widely applicable information criterion", [WAT2010]_), which is simply the score of the parameter combination with the highest likelihood minus the variance of the scores within the fully-converged posterior. Ideally, one prefers models with the fewest free parameters, the WAIC estimates the *effective* number of free parameters for a given model and adjusts the score accordingly. In principle, two models with the same score for their best fits may have wildly different WAIC scores depending on the distribution of scores within their posteriors. This criterion is less sensitive to overfitting than simply comparing the best scores yielded by two models, and should also provide a fair comparisson between models with different numbers of free parameters.

.. [WAT2010] `Watanabe et al. 2010 <http://www.jmlr.org/papers/v11/watanabe10a.html>`_
