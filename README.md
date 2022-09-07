Exact and Approximate Formulas for Bootstrapping Sample Medians
==

Code implementing the formulas in:

https://www.evanmiller.org/bootstrapping-sample-medians.html

Two partial implementations of `DiscreteUnivariateDistribution` are provided:

* `PoissonBootstrapExactSampleMedianDistribution` implements the main formula (a sum of Bessel functions)
* `PoissonBootstrapApproximateSampleMedianDistribution` implements the zeroth-order Bessel formula in the article

Both implement `quantile` and can be sampled. Two-sample bootstrap confidence intervals using these distributions are implemented in bootstrap.jl.
