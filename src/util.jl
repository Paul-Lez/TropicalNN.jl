# Random data generators.

"""
    random_mlp(dims; random_thresholds=false, symbolic=true)

Return random `(weights, biases, thresholds)` for architecture `dims`.
If `random_thresholds=false`, all thresholds are zero. If `symbolic=true`,
convert the generated values to `Rational{BigInt}`.
"""
function random_mlp(dims::AbstractVector{<:Integer}; random_thresholds::Bool = false, symbolic::Bool = true)
    # Convert the Float64 samples to exact rational values.
    if symbolic
        # Use standard deviation sqrt(2 / fan-in).
        weights = [Rational{BigInt}.(sqrt(2/dims[i]) .* Random.randn(dims[i + 1], dims[i]))
                   for i in 1:(length(dims) - 1)]
        biases = [Rational{BigInt}.(sqrt(2/dims[i - 1]) .* Random.randn(dims[i]))
                  for i in 2:length(dims)]
        threshold_range = 2:(length(dims) - 1)
        if random_thresholds
            thresholds = [Rational{BigInt}.(Random.rand(dims[i])) for i in threshold_range]
        else
            thresholds = [Rational{BigInt}.(zeros(dims[i])) for i in threshold_range]
        end
    else
        # Keep the Float64 samples.
        weights = [sqrt(2/dims[i]) .* Random.randn(dims[i + 1], dims[i])
                   for i in 1:(length(dims) - 1)]
        biases = [sqrt(2/dims[i - 1]) .* Random.randn(dims[i]) for i in 2:length(dims)]
        threshold_range = 2:(length(dims) - 1)
        if random_thresholds
            thresholds = [Random.rand(dims[i]) for i in threshold_range]
        else
            thresholds = [zeros(dims[i]) for i in threshold_range]
        end
    end
    return (weights, biases, thresholds)
end

@doc raw"""
    random_signomial(n_vars, n_mons)

Return a random `Signomial` with `n_vars` variables and `n_mons` terms.
Sample coefficients and exponents from a normal distribution with standard
deviation `1 / sqrt(2)`. Convert them to `Rational{BigInt}`.
"""
function random_signomial(n_vars, n_mons)
    scale = 1 / sqrt(2)
    return Signomial(Rational{BigInt}.(scale .* Random.randn(n_mons)),
        [Rational{BigInt}.(scale .* Random.randn(n_vars)) for _ in 1:n_mons];
        sorted = false)
end
