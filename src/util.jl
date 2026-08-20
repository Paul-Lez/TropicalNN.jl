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

"""
    random_maxout_network(dims, pieces, [T=Float64])

Return a random maxout network with architecture `dims`. Each hidden width in
`dims` specifies the number of maxout units in that layer. Each unit has
`pieces` affine inputs. Sample each affine weight and bias from a normal
distribution with standard deviation `sqrt(2 / fan-in)`. Store the samples
with element type `T`.
"""
function random_maxout_network(
        dims::AbstractVector{<:Integer},
        pieces::Integer,
        ::Type{T} = Float64
) where {T}
    length(dims) >= 2 || throw(ArgumentError("The architecture must have an input and output dimension"))
    all(>(0), dims) || throw(ArgumentError("All network dimensions must be positive"))
    pieces > 0 || throw(ArgumentError("The number of maxout pieces must be positive"))

    layers = AbstractNeuralNetworkLayer{T}[]

    # Add the hidden affine maps and their grouped maxout activations.
    for layer_index in 1:(length(dims) - 2)
        fan_in = dims[layer_index]
        width = dims[layer_index + 1]
        scale = sqrt(2 / fan_in)
        push!(layers, AffineLayer(
            T.(scale .* Random.randn(width * pieces, fan_in)),
            T.(scale .* Random.randn(width * pieces))
        ))
        push!(layers, ActivationLayer(maxout(T, pieces), width))
    end

    # Add the final affine output map.
    scale = sqrt(2 / dims[end - 1])
    push!(layers, AffineLayer(
        T.(scale .* Random.randn(dims[end], dims[end - 1])),
        T.(scale .* Random.randn(dims[end]))
    ))
    return NeuralNetwork(layers)
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
