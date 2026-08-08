# Convert an MLP to a tropical rational map.

"""
    single_to_trop(A, b, t)

Convert `x -> max.(A * x + b, t)` to tropical rational functions.

The lengths of `b` and `t` must equal `size(A, 1)`.
"""
function single_to_trop(A::Matrix{T}, b::AbstractVector,
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
    for i in axes(A, 1)
        # Split row i into positive and negative parts.
        pos = Vector{T}()
        neg = Vector{T}()
        for j in axes(A, 2)
            push!(pos, max(A[i, j], 0))
            push!(neg, max(-A[i, j], 0))
        end
        # The numerator is max(b[i] + pos⋅x, t[i] + neg⋅x).
        num = SignomialMonomial(b[i], pos) + SignomialMonomial(t[i], neg)
        # The denominator is neg⋅x.
        den = SignomialMonomial(one(t[i]), neg)
        push!(G, num/den)
    end
    return G
end

"""
    affine_to_trop(A, b)

Convert `x -> A * x + b` to tropical rational functions.

The length of `b` must equal `size(A, 1)`.
"""
function affine_to_trop(A::Matrix{T},
        b::AbstractVector) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    G = RationalSignomial[]

    if size(A, 1) != length(b)
        throw(DimensionMismatch(
            "Dimension mismatch: A has $(size(A,1)) rows, b has length $(length(b)). They must match."
        ))
    end

    R = tropical_semiring(max)
    b = [R(Rational{BigInt}(i)) for i in b]
    sizehint!(G, size(A, 1))
    for i in axes(A, 1)
        pos = Vector{T}()
        neg = Vector{T}()
        for j in axes(A, 2)
            push!(pos, max(A[i, j], 0))
            push!(neg, max(-A[i, j], 0))
        end
        num = SignomialMonomial(b[i], pos)
        den = SignomialMonomial(one(b[i]), neg)
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
    hidden_layer_count = length(linear_maps) - 1
    actual_thresholds = if thresholds === nothing
        [zeros(eltype(linear_maps[k]), size(linear_maps[k], 1))
         for k in 1:hidden_layer_count]
    else
        length(thresholds) == hidden_layer_count || throw(DimensionMismatch(
            "Got $(length(linear_maps)) weight matrices and $(length(thresholds)) " *
            "threshold vectors; expected $hidden_layer_count threshold vectors"
        ))
        thresholds
    end

    layers = Vector{Vector{RationalSignomial}}(undef, length(linear_maps))
    for k in Base.eachindex(linear_maps)
        A = linear_maps[k]
        b = biases[k]
        size(A, 1) == length(b) || throw(DimensionMismatch(
            "Layer $k has $(size(A, 1)) outputs but bias length $(length(b))"
        ))
        if k > firstindex(linear_maps)
            expected_input_dim = size(linear_maps[k - 1], 1)
            size(A, 2) == expected_input_dim || throw(DimensionMismatch(
                "Layer $k has input dimension $(size(A, 2)), but layer $(k - 1) " *
                "has output dimension $expected_input_dim"
            ))
        end

        if k == lastindex(linear_maps)
            layers[k] = RationalSignomial[affine_to_trop(Matrix(A), b)...]
        else
            threshold = actual_thresholds[k]
            size(A, 1) == length(threshold) || throw(DimensionMismatch(
                "Layer $k has $(size(A, 1)) outputs but threshold length " *
                "$(length(threshold))"
            ))
            layers[k] = RationalSignomial[
                single_to_trop(Matrix(A), b, threshold)...
            ]
        end
    end
    return layers
end

"""
    tropicalize(linear_maps, bias, thresholds;
                quicksum=false, strong_elim=false, dedup=false,
                elim_mode=OscarMode(), workers=nothing)

Convert an MLP with an affine output layer to tropical rational functions.
Each hidden layer applies `max.(z, thresholds[i])`. Omit `thresholds` to use
ReLU. The options control batched sums, pruning, and deduplication.

`mlp_to_trop` is a deprecated alias for this function.
"""
function tropicalize(linear_maps::Vector{Matrix{T}}, bias,
        thresholds::Union{AbstractVector{<:AbstractVector}, Nothing} = nothing;
        quicksum::Bool = false, strong_elim::Bool = false,
        dedup::Bool = false,
        elim_mode::LinearRegionsCalculationMode = OscarMode(),
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    layers = _mlp_to_tropical_layers(linear_maps, bias, thresholds)
    output = RationalSignomial[]

    # Compose the layers in network order.
    for (i, ith_tropical) in pairs(layers)
        if i == 1
            output = ith_tropical
        else
            output = comp(ith_tropical, output; quicksum = quicksum)

            if strong_elim
                output = prune(output; mode = elim_mode, workers = workers)
            end
        end
        if dedup
            output = dedup_monomials(output)
        end
    end

    return output
end
