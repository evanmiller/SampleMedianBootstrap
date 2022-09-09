# Formulas from https://www.evanmiller.org/bootstrapping-sample-medians.html
using Distributions
using SpecialFunctions

import Distributions: cdf, pdf, quantile

function exact_bessel_dist_prob(i, n)
    arg = 2*sqrt((n-i)*(i-1))
    sum = 0
    if arg == 0
        # Edge case, i=1 or i=n
        # Compute probability that exposure=(n-1) yields fewer events than
        # exposure=(1)
        poisson = Poisson(n-1)
        for k in 1:20
            delta = (0.5*pdf(poisson, k) + cdf(poisson, k-1))/factorial(k)
            sum += delta
        end
        return exp(-1)*sum
    end
    term = 1 # 1/0!
    for eta in 1:20 # eta=0 is handled at the end
        prefix = ((n-i)/(i-1))^(eta/2)+((i-1)/(n-i))^(eta/2)
        term += 1/factorial(eta)
        delta = (0.5 / factorial(eta) + exp(1) - term)*prefix*besselix(eta, arg)
        sum += delta
    end
    return exp(arg-n)*((exp(1)-1)*besselix(0, arg)+sum)
end

struct PoissonBootstrapExactSampleMedianDistribution <: DiscreteUnivariateDistribution
    n::Int
    cdf::Vector{Float64}
    function PoissonBootstrapExactSampleMedianDistribution(n)
        # Pre-compute the CDFs since there's not a clean formula (to my knowledge)
        cdf = zeros(n)
        # There is a small probability that none of the indexes will be chosen;
        # to ensure the probabilities add to 1, divide by 1 - exp(-n) (note
        # that this divisor rounds to 1 in 64-bit floating point for n > 40)
        sum = 0
        for i in 1:n
            sum += exact_bessel_dist_prob(i, n) / (1 - exp(-n))
            cdf[i] = min(1, sum)
        end
        new(n, cdf)
    end
end

function pdf(dist::PoissonBootstrapExactSampleMedianDistribution, i::Int64)
    return exact_bessel_dist_prob(i, dist.n)
end

function cdf(dist::PoissonBootstrapExactSampleMedianDistribution, i::Int64)
    return dist.cdf[i]
end

function quantile(dist::PoissonBootstrapExactSampleMedianDistribution, q::Real)
    return if q > dist.cdf[dist.n]
        n
    else
        searchsortedfirst(dist.cdf, q)
    end
end

# This is the "double dip" model - Add one exposure to each side of the proposed median
# Pr(y_i=τ) ≅ 2*exp(-(n-1))*I₀(2√((n+1-i)(i)))
function approx_bessel_dist_prob(i::Int64, n::Int64)
    arg = 2*sqrt((n+1-i)*(i))
    return 2*exp(arg-n-1)*(besselix(0, arg))
end

struct PoissonBootstrapApproximateSampleMedianDistribution <: DiscreteUnivariateDistribution
    n::Int
    cdf::Vector{Float64}
    function PoissonBootstrapApproximateSampleMedianDistribution(n)
        cdf = zeros(n)
        sum = 0
        for i in 1:n
            sum += approx_bessel_dist_prob(i, n) / (1 - exp(-n))
            cdf[i] = min(1, sum)
        end
        new(n, cdf)
    end
end

function pdf(dist::PoissonBootstrapApproximateSampleMedianDistribution, i::Int64)
    return approx_bessel_dist_prob(i, dist.n)
end

function cdf(dist::PoissonBootstrapApproximateSampleMedianDistribution, i::Int64)
    return dist.cdf[i]
end

function quantile(dist::PoissonBootstrapApproximateSampleMedianDistribution, q::Real)
    return if q > dist.cdf[dist.n]
        n
    else
        searchsortedfirst(dist.cdf, q)
    end
end

