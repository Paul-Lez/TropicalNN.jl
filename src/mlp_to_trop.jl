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
    mlp_to_trop(linear_maps, bias, thresholds;
                quicksum=false, strong_elim=false, dedup=false,
                elim_mode=OscarMode(), workers=nothing)

Convert an MLP with an affine output layer to tropical rational functions.
Each hidden layer applies `max.(z, thresholds[i])`. Omit `thresholds` to use
ReLU. The options control batched sums, pruning, and deduplication.
"""
function mlp_to_trop(linear_maps::Vector{Matrix{T}}, bias,
        thresholds::Union{AbstractVector{<:AbstractVector}, Nothing} = nothing;
        quicksum::Bool = false, strong_elim::Bool = false,
        dedup::Bool = false,
        elim_mode::LinearRegionsCalculationMode = OscarMode(),
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    if isempty(linear_maps)
        throw(ArgumentError("mlp_to_trop requires at least one layer"))
    end
    if length(bias) != length(linear_maps)
        throw(DimensionMismatch(
            "Dimension mismatch: got $(length(linear_maps)) weight matrices and $(length(bias)) bias vectors. These lengths must match."
        ))
    end
    expected_thresholds = length(linear_maps) - 1
    # Use zero thresholds when the caller omits them.
    if thresholds === nothing
        thresholds = [zeros(T, size(linear_maps[i], 1)) for i in 1:expected_thresholds]
    elseif length(thresholds) != expected_thresholds
        throw(DimensionMismatch(
            "Dimension mismatch: got $(length(linear_maps)) weight matrices and $(length(thresholds)) threshold vectors."
        ))
    end

    output = RationalSignomial[]

    # Compose the layers in network order.
    for i in Base.eachindex(linear_maps)
        A = linear_maps[i]
        b = bias[i]

        # Check the output dimension.
        if size(A, 1) != length(b)
            throw(
                DimensionMismatch(
                "Layer $i: dimension mismatch. A has $(size(A,1)) rows, b has length $(length(b)). They must match.",
            ),
            )
        end

        # Hidden layers apply apply `max(threshold, . )`. The final layer is affine.
        if i == lastindex(linear_maps)
            ith_tropical = affine_to_trop(A, b)
        else
            t = thresholds[i]
            if size(A, 1) != length(t)
                throw(
                    DimensionMismatch(
                    "Layer $i: dimension mismatch. A has $(size(A,1)) rows, t has length $(length(t)). They must match.",
                ),
                )
            end
            ith_tropical = single_to_trop(A, b, t)
        end

        if i == 1
            output = ith_tropical
        else
            output = quicksum ? comp_with_quicksum(ith_tropical, output) :
                     comp(ith_tropical, output)

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
