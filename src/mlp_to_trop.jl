# This file contains functions to convert a multilayer perceptron to a tropical Puiseux rational function

"""
    single_to_trop(A, b, t)

Convert a single ReLU layer to tropical Puiseux rational functions.

# Arguments
- `A::Matrix{T}`: Weight matrix
- `b::AbstractVector`: Bias vector
- `t::AbstractVector`: Activation threshold vector

# Returns
- `Vector{TropicalPuiseuxRational{T}}`: Tropical representation of max(Ax+b, t)

# Throws
- `DimensionMismatch`: If dimensions don't match (A has size(A,1) rows, b and t must have the same length)
"""
function single_to_trop(A::Matrix{T}, b::AbstractVector, t::AbstractVector) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    G = Vector{TropicalPuiseuxRational{T}}()

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
        num = TropicalPuiseuxMonomial(b[i], pos) + TropicalPuiseuxMonomial(t[i], neg)
        # the denominator is the monomila given by the negative part, with coeff the tropical multiplicative 
        # unit, i.e. 0
        den = TropicalPuiseuxMonomial(one(t[i]), neg)
        push!(G, num/den) 
    end 
    return G
end     

"""
    mlp_to_trop(linear_maps, bias, thresholds; quicksum=false, strong_elim=false, dedup=false)

Convert a ReLU multilayer perceptron to a tropical Puiseux rational function.

This function converts a neural network with ReLU-like activation functions (max(x,t))
into an exact tropical geometric representation, enabling analysis of linear regions
and network expressivity.

# Arguments
- `linear_maps::Vector{Matrix{T}}`: Weight matrices for each layer
- `bias`: Bias vectors for each layer
- `thresholds`: Activation threshold vectors for each layer (standard ReLU uses zeros)

where `T<:Union{Oscar.scalar_types, Rational{BigInt}}`

# Keyword Arguments
- `quicksum::Bool=false`: Use faster but less accurate quicksum operations for composition
- `strong_elim::Bool=false`: Apply monomial elimination to remove non-full-dimensional polyhedra at each layer
- `dedup::Bool=false`: Apply deduplication to remove duplicate monomials at each layer

# Returns
- `Vector{TropicalPuiseuxRational{T}}`: Tropical rational functions representing the MLP outputs

# Throws
- `DimensionMismatch`: If matrix/vector dimensions don't match at any layer

# Performance Notes
- `quicksum=true`: Faster for large networks (>3 layers, >10 neurons), defers sorting
- `strong_elim=true`: Reduces complexity by removing redundant monomials, but adds computational overhead
- `dedup=true`: Removes duplicate monomials, useful when composition creates duplicates

# Examples
```julia
# Convert a random 2-3-1 MLP (standard)
W, b, t = random_mlp([2, 3, 1])
f = mlp_to_trop(W, b, t)

# Convert with quicksum for better performance
f_fast = mlp_to_trop(W, b, t, quicksum=true)

# Convert with monomial elimination for reduced complexity
f_reduced = mlp_to_trop(W, b, t, strong_elim=true)

# Combine options for large networks
f_optimized = mlp_to_trop(W, b, t, quicksum=true, strong_elim=true)

# Analyze linear regions
regions = enum_linear_regions_rat(f[1])
```

# See Also
- `single_to_trop`: Convert a single layer
- `comp`, `comp_with_quicksum`: Composition functions
- `monomial_strong_elim`, `dedup_monomials`: Simplification functions
"""
function mlp_to_trop(linear_maps::Vector{Matrix{T}}, bias, thresholds;
                      quicksum::Bool=false, strong_elim::Bool=false, dedup::Bool=false) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    R = tropical_semiring(max)

    # Initialisation: the first vector of tropical rational functions
    output = single_to_trop(linear_maps[1], bias[1], thresholds[1])

    # Apply initial simplification if requested
    if dedup
        output = dedup_monomials(output)
    end

    # Iterate through the layers and compose variable output with the current layer at each step
    for i in Base.eachindex(linear_maps)
        A = linear_maps[i]
        b = bias[i]
        t = thresholds[i]

        # Check dimensions match
        if size(A, 1) != length(b) || size(A, 1) != length(t)
            throw(DimensionMismatch(
                "Layer $i: dimension mismatch. A has $(size(A,1)) rows, b has length $(length(b)), t has length $(length(t)). All must match."
            ))
        end

        if i != 1
            # Compute the vector of tropical rational functions corresponding to max(Ax+b, t)
            ith_tropical = single_to_trop(A, b, t)

            # Compose with the output of the previous layer
            output = quicksum ? comp_with_quicksum(ith_tropical, output) : comp(ith_tropical, output)

            # Apply simplification if requested
            if strong_elim
                output = monomial_strong_elim(output)
            end
            if dedup
                output = dedup_monomials(output)
            end
        end
    end

    return output
end 

"""
    mlp_to_trop_with_quicksum(linear_maps, bias, thresholds)

**DEPRECATED**: Use `mlp_to_trop(linear_maps, bias, thresholds, quicksum=true)` instead.

Computes the tropical Puiseux rational function associated to a multilayer perceptron
using quicksum operations for tropical objects.
"""
function mlp_to_trop_with_quicksum(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    @warn "mlp_to_trop_with_quicksum is deprecated, use mlp_to_trop(..., quicksum=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, quicksum=true)
end 

"""
    mlp_to_trop_with_mul_with_quicksum(linear_maps, bias, thresholds)

**DEPRECATED**: Use `mlp_to_trop(linear_maps, bias, thresholds, quicksum=true)` instead.

Computes the tropical Puiseux rational function associated to a multilayer perceptron
using mul_with_quicksum version of multiplication for tropical objects.
"""
function mlp_to_trop_with_mul_with_quicksum(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    @warn "mlp_to_trop_with_mul_with_quicksum is deprecated, use mlp_to_trop(..., quicksum=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, quicksum=true)
end 

"""
    random_mlp(dims; random_thresholds=false, symbolic=true)

Generate a random multilayer perceptron with specified architecture.

# Arguments
- `dims::AbstractVector{<:Integer}`: Array of integers specifying the width of each layer (e.g., [2, 3, 1] for 2 inputs, 3 hidden neurons, 1 output)

# Keyword Arguments
- `random_thresholds::Bool=false`: If true, activation thresholds are chosen randomly. If false, all thresholds are 0 (standard ReLU)
- `symbolic::Bool=true`: If true, use exact rational arithmetic (Rational{BigInt}). If false, use floating point

# Returns
- `Tuple{Vector{Matrix}, Vector{Vector}, Vector{Vector}}`: (weights, biases, thresholds) for the MLP

# Examples
```julia
# Create a 2-3-1 MLP with ReLU activations
W, b, t = random_mlp([2, 3, 1])

# Create with random thresholds using floating point
W, b, t = random_mlp([2, 4, 1], random_thresholds=true, symbolic=false)
```
"""
function random_mlp(dims::AbstractVector{<:Integer}; random_thresholds::Bool=false, symbolic::Bool=true)
    # if symbolic is set to true then we work with symbolic fractions. 
    if symbolic 
        # Use He initialisation, i.e. we sample weights with distribution N(0, sqrt(2/n))
        weights = [Rational{BigInt}.(rand(Normal(0, sqrt(2/dims[1])), dims[i+1], dims[i])) for i in 1:length(dims)-1]
        biases = [Rational{BigInt}.(rand(Normal(0, sqrt(2/dims[1])), dims[i])) for i in 2:length(dims)]
        if random_thresholds
            thresholds = [Rational{BigInt}.(rand(dims[i])) for i in 2:length(dims)]
        else 
            thresholds = [Rational{BigInt}.(zeros(dims[i])) for i in 2:length(dims)]
        end 
    else # otherwise we work with Floats
        # Use He initialisation, i.e. we sample weights with distribution N(0, sqrt(2/n))
        weights = [rand(Normal(0, sqrt(2/dims[1])), dims[i+1], dims[i]) for i in 1:length(dims)-1]
        biases = [rand(Normal(0, sqrt(2/dims[1])), dims[i]) for i in 2:length(dims)]
        if random_thresholds
            thresholds = [rand(dims[i]) for i in 2:length(dims)]
        else 
            thresholds = [zeros(dims[i]) for i in 2:length(dims)]
        end 
    end 
    return (weights, biases, thresholds)
end 

@doc raw"""
    random_pmap(n_vars,n_mons)

Returns a random tropical polynomial in `n_vars` variables with `n_mons` monomials.
"""
function random_pmap(n_vars,n_mons)
    return TropicalPuiseuxPoly(Rational{BigInt}.(rand(Normal(0,1/sqrt(2)),n_mons)),[Rational{BigInt}.(rand(Normal(0,1/sqrt(2)),n_vars)) for _ in 1:n_mons],true)
end

"""
    mlp_to_trop_with_dedup(linear_maps, bias, thresholds)

**DEPRECATED**: Use `mlp_to_trop(linear_maps, bias, thresholds, dedup=true)` instead.

Computes the tropical Puiseux rational function associated to a multilayer perceptron.
Runs a deduplication function to remove duplicate monomials at each layer.
"""
function mlp_to_trop_with_dedup(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    @warn "mlp_to_trop_with_dedup is deprecated, use mlp_to_trop(..., dedup=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, dedup=true)
end 