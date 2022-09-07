using Random
using Statistics
using Distributions
include("bootstrap.jl")

genNorm1 = Normal(1.1,1)
genNorm2 = Normal(1.2,1)
alpha = 0.05
N1 = 1000
N2 = 1000
B = 10000

x = rand(genNorm1, N1)
y = rand(genNorm2, N2)

@show two_sample_poisson_bootstrap_binomial_quantile_confidence_interval(x, y, alpha, 0.5, B)
@show two_sample_poisson_bootstrap_approximate_median_confidence_interval(x, y, alpha, B)
@show two_sample_poisson_bootstrap_exact_median_confidence_interval(x, y, alpha, B)
