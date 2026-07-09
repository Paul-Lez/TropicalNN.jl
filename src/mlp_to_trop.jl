# This file contains functions to convert a multilayer perceptron to a tropical Puiseux rational function

"""
    single_to_trop(A, b, t)

Convert a single ReLU layer to tropical Puiseux rational functions.

# Arguments
- `A::Matrix{T}`: Weight matrix
- `b::AbstractVector`: Bias vector
- `t::AbstractVector`: Activation threshold vector

# Returns
- `Vector{RationalSignomial{T}}`: Tropical representation of max(Ax+b, t)

# Throws
- `DimensionMismatch`: If dimensions don't match (A has size(A,1) rows, b and t must have the same length)
"""
function single_to_trop(A::Matrix{T}, b::AbstractVector,
        t::AbstractVector) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    G = RationalSignomial[]

    # Check dimensions match
    if size(A, 1) != length(b) || size(A, 1) != length(t)
        throw(DimensionMismatch(
            "Dimension mismatch: A has $(size(A,1)) rows, b has length $(length(b)), t has length $(length(t)). All must match."
        ))
    end
    R = tropical_semiring(max)
    # first make sure that the entries of b are elements of the tropical semiring
    b = [R(Rational(i)) for i in b]
    # and same for t
    t = [R(Rational(i)) for i in t]
    sizehint!(G, size(A, 1))
    for i in axes(A, 1)
        # first split the i-th line of A into its positive and negative components
        pos = Vector{T}()
        neg = Vector{T}()
        for j in axes(A, 2)
            push!(pos, max(A[i, j], 0))
            push!(neg, max(-A[i, j], 0))
        end
        # the numerator is the monomial given by the positive part, with coeff b[i], plus the monomial given by the negative part
        # with coeff t[i]
        num = SignomialMonomial(b[i], pos) + SignomialMonomial(t[i], neg)
        # the denominator is the monomial given by the negative part, with coeff the tropical multiplicative 
        # unit, i.e. 0
        den = SignomialMonomial(one(t[i]), neg)
        push!(G, num/den)
    end
    return G
end

"""
    affine_to_trop(A, b)

Convert a single affine layer to tropical Puiseux rational functions.
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
    b = [R(Rational(i)) for i in b]
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
    mlp_to_trop(linear_maps, bias, thresholds; quicksum=false, strong_elim=false, dedup=false)

Convert a ReLU MLP with affine output layer to tropical rational functions,
one per output neuron. `linear_maps`, `bias`, and `thresholds` are layer-wise
weights, biases, and activation thresholds. `thresholds` has one entry for
each hidden layer, so it must have length `length(linear_maps) - 1`; the final
layer is affine and has no threshold. The vector of thresholds is optional; 
if not provided, all thresholds are set to be zero. 

Options: `quicksum` uses the `quicksum` approach for computing sums; `strong_elim` removes
monomials with non-full-dimensional regions after each layer; `dedup` calls
`dedup_monomials` after each layer. Throws `DimensionMismatch` for inconsistent
layer sizes.
"""
function mlp_to_trop(linear_maps::Vector{Matrix{T}}, bias,
        thresholds::Union{AbstractVector{<:AbstractVector}, Nothing} = nothing;
        quicksum::Bool = false, strong_elim::Bool = false,
        dedup::Bool = false) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    if isempty(linear_maps)
        throw(ArgumentError("mlp_to_trop requires at least one layer"))
    end
    if length(bias) != length(linear_maps)
        throw(DimensionMismatch(
            "Dimension mismatch: got $(length(linear_maps)) weight matrices and $(length(bias)) bias vectors. These lengths must match."
        ))
    end
    expected_thresholds = length(linear_maps) - 1
    # If thresholds aren't provided, then we take them all to be zero.
    if thresholds === nothing
        thresholds = [zeros(T, size(linear_maps[i], 1)) for i in 1:expected_thresholds]
    elseif length(thresholds) != expected_thresholds
        throw(DimensionMismatch(
            "Dimension mismatch: got $(length(linear_maps)) weight matrices and $(length(thresholds)) threshold vectors."
        ))
    end

    output = RationalSignomial[]

    # Iterate through the layers and compose variable output with the current layer at each step
    for i in Base.eachindex(linear_maps)
        A = linear_maps[i]
        b = bias[i]

        # Check dimensions match
        if size(A, 1) != length(b)
            throw(
                DimensionMismatch(
                "Layer $i: dimension mismatch. A has $(size(A,1)) rows, b has length $(length(b)). They must match.",
            ),
            )
        end

        # Hidden layers are ReLU-clamped; the final layer is affine.
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
                output = monomial_strong_elim(output)
            end
        end
        if dedup
            output = dedup_monomials(output)
        end
    end

    return output
end

"""
    mlp_to_trop_with_quicksum(linear_maps, bias, thresholds)

Deprecated; use `mlp_to_trop(linear_maps, bias, thresholds, quicksum=true)`.
"""
function mlp_to_trop_with_quicksum(linear_maps::Vector{Matrix{T}}, bias,
        thresholds) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    @warn "mlp_to_trop_with_quicksum is deprecated, use mlp_to_trop(..., quicksum=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, quicksum = true)
end

"""
    mlp_to_trop_with_dedup(linear_maps, bias, thresholds)

Deprecated; use `mlp_to_trop(linear_maps, bias, thresholds, dedup=true)`.
"""
function mlp_to_trop_with_dedup(linear_maps::Vector{Matrix{T}}, bias,
        thresholds) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    @warn "mlp_to_trop_with_dedup is deprecated, use mlp_to_trop(..., dedup=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, dedup = true)
end
