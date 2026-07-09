# This files contains a few utility functions

"""
    random_mlp(dims; random_thresholds=false, symbolic=true)

Generate random weights, biases, and hidden-layer thresholds for layer widths
`dims`. Returns `(weights, biases, thresholds)`. If `symbolic=true`, entries
are converted to `Rational{BigInt}`; otherwise they are floating-point values.
When `random_thresholds=false`, all hidden-layer thresholds are zero.
"""
function random_mlp(dims::AbstractVector{<:Integer}; random_thresholds::Bool = false, symbolic::Bool = true)
    # if symbolic is set to true then we work with symbolic fractions. 
    if symbolic
        # Use He initialisation: variance 2/fan_in, where fan_in is dims[i] for weights and dims[i-1] for biases
        weights = [Rational{BigInt}.(rand(Normal(0, sqrt(2/dims[i])), dims[i + 1], dims[i]))
                   for i in 1:(length(dims) - 1)]
        biases = [Rational{BigInt}.(rand(Normal(0, sqrt(2/dims[i - 1])), dims[i]))
                  for i in 2:length(dims)]
        threshold_range = 2:(length(dims) - 1)
        if random_thresholds
            thresholds = [Rational{BigInt}.(rand(dims[i])) for i in threshold_range]
        else
            thresholds = [Rational{BigInt}.(zeros(dims[i])) for i in threshold_range]
        end
    else # otherwise we work with Floats
        # Use He initialisation: variance 2/fan_in, where fan_in is dims[i] for weights and dims[i-1] for biases
        weights = [rand(Normal(0, sqrt(2/dims[i])), dims[i + 1], dims[i])
                   for i in 1:(length(dims) - 1)]
        biases = [rand(Normal(0, sqrt(2/dims[i - 1])), dims[i]) for i in 2:length(dims)]
        threshold_range = 2:(length(dims) - 1)
        if random_thresholds
            thresholds = [rand(dims[i]) for i in threshold_range]
        else
            thresholds = [zeros(dims[i]) for i in threshold_range]
        end
    end
    return (weights, biases, thresholds)
end

@doc raw"""
    random_pmap(n_vars, n_mons)

Generate a random tropical polynomial with `n_vars` variables and `n_mons`
monomials. Coefficients and exponents are sampled from `Normal(0, 1/sqrt(2))`
and converted to `Rational{BigInt}`.
"""
function random_pmap(n_vars, n_mons)
    return Signomial(Rational{BigInt}.(rand(Normal(0, 1/sqrt(2)), n_mons)),
        [Rational{BigInt}.(rand(Normal(0, 1/sqrt(2)), n_vars)) for _ in 1:n_mons];
        sorted = false)
end
