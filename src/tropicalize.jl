# Convert an MLP to a tropical rational map.

"""
    single_to_trop(A, b, t)

Convert `x -> max.(A * x + b, t)` to tropical rational functions.

The lengths of `b` and `t` must equal `size(A, 1)`.
The entries of `A` must be Oscar scalars or `Rational{BigInt}` values. The
entries of `b` and `t` must be convertible to `Rational{BigInt}`.
"""
function single_to_trop(A::AbstractMatrix{T}, b::AbstractVector,
        t::AbstractVector) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    G = RationalSignomial[]

    # Check the output dimension.
    if size(A, 1) != length(b) || size(A, 1) != length(t)
        throw(DimensionMismatch(
            "Dimension mismatch: A has $(size(A,1)) rows, b has length $(length(b)), t has length $(length(t)). All must match."
        ))
    end
    R = tropical_semiring(max)
    # Convert the bias and threshold to max-plus scalars.
    b = [R(Rational{BigInt}(i)) for i in b]
    t = [R(Rational{BigInt}(i)) for i in t]
    sizehint!(G, size(A, 1))
    for (row, i) in enumerate(axes(A, 1))
        # Split row i into positive and negative parts.
        pos = Vector{T}()
        neg = Vector{T}()
        for j in axes(A, 2)
            push!(pos, max(A[i, j], 0))
            push!(neg, max(-A[i, j], 0))
        end
        # The numerator is max(b[i] + pos⋅x, t[i] + neg⋅x).
        num = signomial_monomial(b[row], pos) + signomial_monomial(t[row], neg)
        # The denominator is neg⋅x.
        den = signomial_monomial(one(t[row]), neg)
        push!(G, num/den)
    end
    return G
end

"""
    affine_to_trop(A, b)

Convert `x -> A * x + b` to tropical rational functions.

The length of `b` must equal `size(A, 1)`.
Entries of `A` must support comparison with zero and subtraction. Entries of
`b` must be convertible to `Rational{BigInt}`.
"""
function affine_to_trop(A::AbstractMatrix{T}, b::AbstractVector) where {T}
    G = RationalSignomial[]

    if size(A, 1) != length(b)
        throw(DimensionMismatch(
            "Dimension mismatch: A has $(size(A,1)) rows, b has length $(length(b)). They must match."
        ))
    end

    R = tropical_semiring(max)
    b = [R(Rational{BigInt}(i)) for i in b]
    sizehint!(G, size(A, 1))
    # Split each row into numerator and denominator exponents.
    for (row, i) in enumerate(axes(A, 1))
        pos = Vector{T}()
        neg = Vector{T}()
        for j in axes(A, 2)
            positive_entry = max(A[i, j], zero(T))
            push!(pos, positive_entry)
            push!(neg, positive_entry - A[i, j])
        end
        num = signomial_monomial(b[row], pos)
        den = signomial_monomial(one(b[row]), neg)
        push!(G, num/den)
    end
    return G
end

"""
    _mlp_to_tropical_layers(linear_maps, biases, thresholds=nothing)

Convert each MLP layer to a vector of rational signomials without composing
the layers. Hidden layers apply `max.(A * x + b, threshold)`. The final layer
is affine.
"""
function _mlp_to_tropical_layers(
        linear_maps::AbstractVector{<:AbstractMatrix},
        biases::AbstractVector{<:AbstractVector},
        thresholds::Union{Nothing, AbstractVector{<:AbstractVector}} = nothing
)
    isempty(linear_maps) && throw(ArgumentError(
        "MLP conversion requires at least one layer"
    ))
    length(biases) == length(linear_maps) || throw(DimensionMismatch(
        "Got $(length(linear_maps)) weight matrices and $(length(biases)) bias vectors"
    ))
    map_indices = collect(eachindex(linear_maps))
    bias_indices = collect(eachindex(biases))
    hidden_layer_count = length(linear_maps) - 1
    # Use zero thresholds for ReLU when the caller omits them.
    actual_thresholds = if thresholds === nothing
        [zeros(eltype(linear_maps[map_indices[k]]), size(linear_maps[map_indices[k]], 1))
         for k in 1:hidden_layer_count]
    else
        length(thresholds) == hidden_layer_count || throw(DimensionMismatch(
            "Got $(length(linear_maps)) weight matrices and $(length(thresholds)) " *
            "threshold vectors; expected $hidden_layer_count threshold vectors"
        ))
        thresholds
    end
    threshold_indices = thresholds === nothing ? nothing : collect(eachindex(thresholds))

    layers = Vector{Vector{RationalSignomial}}(undef, length(linear_maps))
    # Convert each layer before composition changes its input variables.
    for (layer, map_index) in enumerate(map_indices)
        A = linear_maps[map_index]
        b = biases[bias_indices[layer]]
        size(A, 1) == length(b) || throw(DimensionMismatch(
            "Layer $layer has $(size(A, 1)) outputs but bias length $(length(b))"
        ))
        if layer > 1
            expected_input_dim = size(linear_maps[map_indices[layer - 1]], 1)
            size(A, 2) == expected_input_dim || throw(DimensionMismatch(
                "Layer $layer has input dimension $(size(A, 2)), but layer $(layer - 1) " *
                "has output dimension $expected_input_dim"
            ))
        end

        # Keep the output affine and apply thresholds only to hidden layers.
        if layer == length(map_indices)
            layers[layer] = RationalSignomial[affine_to_trop(A, b)...]
        else
            threshold = thresholds === nothing ? actual_thresholds[layer] :
                        thresholds[threshold_indices[layer]]
            size(A, 1) == length(threshold) || throw(DimensionMismatch(
                "Layer $layer has $(size(A, 1)) outputs but threshold length " *
                "$(length(threshold))"
            ))
            layers[layer] = RationalSignomial[single_to_trop(A, b, threshold)...]
        end
    end
    return layers
end

"""
    tropicalize_layers(layer::AbstractNeuralNetworkLayer)

Return the uncomposed tropical rational map for each atomic layer in `layer`.
Each element is a vector-valued map. Nested networks are flattened.
"""
function tropicalize_layers(layer::AffineLayer)
    return [affine_to_trop(layer.weight, layer.bias)]
end

function tropicalize_layers(layer::ActivationLayer)
    total_input_dimension = input_dimension(layer)
    outputs = RationalSignomial[]
    sizehint!(outputs, output_dimension(layer))
    first_input = 1
    # Embed each activation block in the full layer input space.
    for activation in layer.activations
        activation_dimension = nvars(activation)
        last_input = first_input + activation_dimension - 1
        push!(outputs, _embed_variables(
            activation,
            first_input:last_input,
            total_input_dimension
        ))
        first_input = last_input + 1
    end
    return [outputs]
end

function tropicalize_layers(network::NeuralNetwork)
    layers = Vector{Vector{RationalSignomial}}()
    # Flatten nested networks and preserve the layer order.
    for layer in network
        append!(layers, tropicalize_layers(layer))
    end
    return layers
end

function _compose_tropical_layers(
        layers::AbstractVector{<:AbstractVector{<:RationalSignomial}};
        quicksum::Bool = false,
        prune::Bool = false,
        dedup::Bool = false,
        elim_mode::LinearRegionsCalculationMode = OscarMode(),
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    output = RationalSignomial[]

    for (layer_index, tropical_layer) in pairs(layers)
        # The first layer already uses the network input variables.
        if layer_index == firstindex(layers)
            output = RationalSignomial[tropical_layer...]
        else
            output = comp(tropical_layer, output; quicksum = quicksum)

            # Prune only after the layer composition is complete.
            if prune
                output = TropicalNN.prune(output; mode = elim_mode, workers = workers)
            end
        end
        # Remove tropical-zero terms before the next composition.
        if dedup
            output = [RationalSignomial(
                          _remove_zero_matrix_terms(f.num),
                          _remove_zero_matrix_terms(f.den)
                      ) for f in output]
        end
    end

    return output
end

"""
    tropicalize(linear_maps, bias, thresholds;
                quicksum=false, prune=false, dedup=false,
                elim_mode=OscarMode(), workers=nothing)

Convert an MLP with an affine output layer to tropical rational functions.
Each hidden layer applies `max.(z, thresholds[i])`. Omit `thresholds` to use
ReLU. The keywords control batched sums and pruning. Set `dedup=true` to remove
terms whose coefficient is tropical zero.
`linear_maps` can be an `AbstractVector` of `AbstractMatrix` values. The bias
and threshold collections can also be abstract vectors.

`mlp_to_trop` is a deprecated alias for this function.
"""
function tropicalize(
        linear_maps::AbstractVector{<:AbstractMatrix{T}},
        bias::AbstractVector{<:AbstractVector},
        thresholds::Union{AbstractVector{<:AbstractVector}, Nothing} = nothing;
        quicksum::Bool = false, prune::Bool = false,
        dedup::Bool = false,
        elim_mode::LinearRegionsCalculationMode = OscarMode(),
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    layers = _mlp_to_tropical_layers(linear_maps, bias, thresholds)
    return _compose_tropical_layers(
        layers;
        quicksum = quicksum,
        prune = prune,
        dedup = dedup,
        elim_mode = elim_mode,
        workers = workers
    )
end

"""
    tropicalize(layer::AbstractNeuralNetworkLayer;
                quicksum=false, prune=false, dedup=false,
                elim_mode=OscarMode(), workers=nothing)

Convert a neural-network layer or network to tropical rational functions. The
keywords control batched sums and pruning. Set `dedup=true` to remove terms
whose coefficient is tropical zero.
"""
function tropicalize(
        layer::AbstractNeuralNetworkLayer{T};
        quicksum::Bool = false,
        prune::Bool = false,
        dedup::Bool = false,
        elim_mode::LinearRegionsCalculationMode = OscarMode(),
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
) where {T}
    return _compose_tropical_layers(
        tropicalize_layers(layer);
        quicksum = quicksum,
        prune = prune,
        dedup = dedup,
        elim_mode = elim_mode,
        workers = workers
    )
end
