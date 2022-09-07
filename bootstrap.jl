# Poisson bootstrap code adapted from
# https://github.com/MSchultzberg/fast_quantile_bootstrap

using Distributions

include("./distribution.jl");

function confidence_interval(
        x::Vector{Float64}, x_quantile_dist::DiscreteUnivariateDistribution,
        y::Vector{Float64}, y_quantile_dist::DiscreteUnivariateDistribution,
        alpha::Real, B::Int)
    sample1_quantile_indexes=rand(x_quantile_dist, B)
    sample2_quantile_indexes=rand(y_quantile_dist, B)
    min_x = minimum(sample1_quantile_indexes)
    max_x = maximum(sample1_quantile_indexes)
    min_y = minimum(sample2_quantile_indexes)
    max_y = maximum(sample2_quantile_indexes)

    # Only need to sort the points that were actually drawn
    ordered_y = partialsort(y, min_y:max_y)
    ordered_x = partialsort(x, min_x:max_x)

    diff_in_quantile = ordered_y[sample2_quantile_indexes .- (min_y - 1)] - ordered_x[sample1_quantile_indexes .- (min_x - 1)]

    return quantile!(diff_in_quantile, [alpha/2,1-alpha/2])
end

function two_sample_poisson_bootstrap_binomial_quantile_confidence_interval(
        x::Vector{Float64},
        y::Vector{Float64},
        alpha::Real, q::Real, B::Int)
    return confidence_interval(x, Binomial(length(x)+1, q), y, Binomial(length(y)+1, q), alpha, B)
end

function two_sample_poisson_bootstrap_approximate_median_confidence_interval(
        x::Vector{Float64},
        y::Vector{Float64},
        alpha::Real, B::Int)
    return confidence_interval(x, PoissonBootstrapApproximateSampleMedianDistribution(length(x)),
                               y, PoissonBootstrapApproximateSampleMedianDistribution(length(y)), alpha, B)
end

function two_sample_poisson_bootstrap_exact_median_confidence_interval(
        x::Vector{Float64},
        y::Vector{Float64},
        alpha::Real, B::Int)
    return confidence_interval(x, PoissonBootstrapExactSampleMedianDistribution(length(x)),
                               y, PoissonBootstrapExactSampleMedianDistribution(length(y)), alpha, B)
end
