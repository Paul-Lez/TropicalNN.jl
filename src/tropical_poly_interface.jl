# Unified interface for tropical Puiseux polynomials
#
# This module provides a unified API for tropical polynomials that automatically
# selects the optimal internal representation based on dimension:
#   - Dimensions 1-5:  StaticArrays (stack-allocated, ~1.4x speedup)
#   - Dimensions >5:   Matrix storage (cache-efficient, ~2-3x speedup)
#
# The user-facing API is dimension-agnostic; implementation details are hidden.

using Oscar
using StaticArrays
using LinearAlgebra

#==============================================================================#
#                         ABSTRACT TYPE HIERARCHY                               #
#==============================================================================#

"""
    AbstractTropicalPuiseuxPoly{T}

Abstract supertype for all tropical Puiseux polynomial representations.
Concrete subtypes implement different storage strategies optimized for
different dimension ranges.

All subtypes must implement:
- `Base.:+(f, g)` - Addition (tropical max)
- `Base.:*(f, g)` - Multiplication
- `eval(f, x)` - Point evaluation
- `nvars(f)` - Number of variables
- `length(f)` - Number of monomials
- `get_exp(f, i)` - Get i-th exponent vector
- `get_coeff(f, exp)` - Get coefficient for exponent
"""
abstract type AbstractTropicalPuiseuxPoly{T} end

"""
    AbstractTropicalPuiseuxRational{T}

Abstract supertype for tropical Puiseux rational functions (quotients of polynomials).
"""
abstract type AbstractTropicalPuiseuxRational{T} end

#==============================================================================#
#                    STATIC ARRAYS IMPLEMENTATION (dim ≤ 5)                    #
#==============================================================================#

"""
    TropicalPuiseuxPolyStatic{T, N}

Tropical Puiseux polynomial using StaticArrays for exponent storage.
Optimal for dimensions 1-5, providing ~1.3-1.5x speedup over vector storage.

# Type Parameters
- `T`: Numeric type for exponents (typically Float64 or Rational)
- `N`: Dimension (number of variables) - known at compile time

# Performance
- Stack-allocated exponent vectors (no heap allocation)
- Efficient comparison and hashing for small tuples
- Best for visualization (2D/3D) and small MLPs
"""
struct TropicalPuiseuxPolyStatic{T, N} <: AbstractTropicalPuiseuxPoly{T}
    coeff::Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}
    exp::Vector{SVector{N,T}}

    function TropicalPuiseuxPolyStatic{T,N}(
        coeff::Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp::Vector{SVector{N,T}}
    ) where {T,N}
        new{T,N}(coeff, exp)
    end
end

# Constructor from vector-of-vectors
function TropicalPuiseuxPolyStatic{T,N}(
    coeff_dict::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
    exp_vecs::Vector{Vector{T}},
    sorted::Bool=false
) where {T,N}
    # Convert to static vectors
    exp_static = [SVector{N,T}(e) for e in exp_vecs]
    coeff_static = Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    for (k, v) in coeff_dict
        coeff_static[SVector{N,T}(k)] = v
    end

    # Sort if needed
    if !sorted
        sort!(exp_static)
    end

    return TropicalPuiseuxPolyStatic{T,N}(coeff_static, exp_static)
end

# Constructor from coefficient vector and exponent vector
function TropicalPuiseuxPolyStatic{T,N}(
    coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
    exp_vecs::Vector{Vector{T}},
    sorted::Bool=false
) where {T,N}
    exp_static = [SVector{N,T}(e) for e in exp_vecs]

    if !sorted
        perm = sortperm(exp_static)
        exp_static = exp_static[perm]
        coeffs = coeffs[perm]
    end

    coeff_static = Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}(
        exp_static[i] => coeffs[i] for i in eachindex(coeffs)
    )

    return TropicalPuiseuxPolyStatic{T,N}(coeff_static, exp_static)
end

#==============================================================================#
#                    MATRIX IMPLEMENTATION (dim > 5)                           #
#==============================================================================#

"""
    TropicalPuiseuxPolyMatrix{T}

Tropical Puiseux polynomial using matrix-based exponent storage.
Optimal for dimensions >5, providing ~1.5-3x speedup over vector storage.

# Fields
- `exp::Matrix{T}`: Exponent matrix (dim × n_monomials), each column is one exponent
- `coeff::Vector{TropicalSemiringElem}`: Coefficients parallel to exp columns
- `dim::Int`: Dimension (number of variables)

# Performance
- Excellent cache locality from contiguous memory
- No per-vector allocation overhead
- Better for medium/large MLPs (10-100+ input dimensions)
"""
struct TropicalPuiseuxPolyMatrix{T} <: AbstractTropicalPuiseuxPoly{T}
    exp::Matrix{T}
    coeff::Vector{Oscar.TropicalSemiringElem{typeof(max)}}
    dim::Int

    function TropicalPuiseuxPolyMatrix{T}(
        exp::Matrix{T},
        coeff::Vector{Oscar.TropicalSemiringElem{typeof(max)}}
    ) where T
        dim, n_monomials = size(exp)
        @assert length(coeff) == n_monomials "Coefficient count must match monomial count"
        new{T}(exp, coeff, dim)
    end
end

# Constructor from vector-of-vectors
function TropicalPuiseuxPolyMatrix{T}(
    coeff_dict::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
    exp_vecs::Vector{Vector{T}},
    sorted::Bool=false
) where T
    if isempty(exp_vecs)
        return TropicalPuiseuxPolyMatrix{T}(Matrix{T}(undef, 0, 0), eltype(values(coeff_dict))[])
    end

    dim = length(exp_vecs[1])
    n_monomials = length(exp_vecs)

    # Sort if needed
    if !sorted
        exp_vecs = sort(exp_vecs)
    end

    # Convert to matrix
    exp_matrix = Matrix{T}(undef, dim, n_monomials)
    coeff_vec = Vector{Oscar.TropicalSemiringElem{typeof(max)}}(undef, n_monomials)

    @inbounds for i in 1:n_monomials
        for d in 1:dim
            exp_matrix[d, i] = exp_vecs[i][d]
        end
        coeff_vec[i] = coeff_dict[exp_vecs[i]]
    end

    return TropicalPuiseuxPolyMatrix{T}(exp_matrix, coeff_vec)
end

# Constructor from coefficient vector and exponent vector
function TropicalPuiseuxPolyMatrix{T}(
    coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
    exp_vecs::Vector{Vector{T}},
    sorted::Bool=false
) where T
    if isempty(exp_vecs)
        return TropicalPuiseuxPolyMatrix{T}(Matrix{T}(undef, 0, 0), coeffs)
    end

    dim = length(exp_vecs[1])
    n_monomials = length(exp_vecs)

    if !sorted
        perm = sortperm(exp_vecs)
        exp_vecs = exp_vecs[perm]
        coeffs = coeffs[perm]
    end

    exp_matrix = Matrix{T}(undef, dim, n_monomials)
    @inbounds for i in 1:n_monomials
        for d in 1:dim
            exp_matrix[d, i] = exp_vecs[i][d]
        end
    end

    return TropicalPuiseuxPolyMatrix{T}(exp_matrix, coeffs)
end

#==============================================================================#
#                         COMMON INTERFACE METHODS                              #
#==============================================================================#

# Number of variables
Oscar.nvars(f::TropicalPuiseuxPolyStatic{T,N}) where {T,N} = N
Oscar.nvars(f::TropicalPuiseuxPolyMatrix) = f.dim

# Number of monomials
Base.length(f::TropicalPuiseuxPolyStatic) = length(f.exp)
Base.length(f::TropicalPuiseuxPolyMatrix) = size(f.exp, 2)

# Iteration
Base.eachindex(f::AbstractTropicalPuiseuxPoly) = Base.OneTo(length(f))

# Get exponent vector (returns view for matrix, copy for static)
function get_exp(f::TropicalPuiseuxPolyStatic{T,N}, i::Int) where {T,N}
    return f.exp[i]
end

function get_exp(f::TropicalPuiseuxPolyMatrix, i::Int)
    return @view f.exp[:, i]
end

# Get coefficient
function get_coeff(f::TropicalPuiseuxPolyStatic, i::Int)
    return f.coeff[f.exp[i]]
end

function get_coeff(f::TropicalPuiseuxPolyMatrix, i::Int)
    return f.coeff[i]
end

#==============================================================================#
#                    ARITHMETIC: STATIC IMPLEMENTATION                          #
#==============================================================================#

function Base.:+(f::TropicalPuiseuxPolyStatic{T,N}, g::TropicalPuiseuxPolyStatic{T,N}) where {T,N}
    lf, lg = length(f.exp), length(g.exp)

    h_coeff = Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(h_coeff, lf + lg)
    h_exp = Vector{SVector{N,T}}()
    sizehint!(h_exp, lf + lg)

    trop_zero = zero(first(values(f.coeff)))

    # Merge sorted lists
    i, j = 1, 1
    @inbounds while i <= lf && j <= lg
        f_exp, g_exp = f.exp[i], g.exp[j]
        if f_exp < g_exp
            c = f.coeff[f_exp]
            if c != trop_zero
                h_coeff[f_exp] = c
                push!(h_exp, f_exp)
            end
            i += 1
        elseif g_exp < f_exp
            c = g.coeff[g_exp]
            if c != trop_zero
                h_coeff[g_exp] = c
                push!(h_exp, g_exp)
            end
            j += 1
        else  # equal
            c = f.coeff[f_exp] + g.coeff[g_exp]
            if c != trop_zero
                h_coeff[f_exp] = c
                push!(h_exp, f_exp)
            end
            i += 1
            j += 1
        end
    end

    # Remaining from f
    @inbounds while i <= lf
        f_exp = f.exp[i]
        c = f.coeff[f_exp]
        if c != trop_zero
            h_coeff[f_exp] = c
            push!(h_exp, f_exp)
        end
        i += 1
    end

    # Remaining from g
    @inbounds while j <= lg
        g_exp = g.exp[j]
        c = g.coeff[g_exp]
        if c != trop_zero
            h_coeff[g_exp] = c
            push!(h_exp, g_exp)
        end
        j += 1
    end

    return TropicalPuiseuxPolyStatic{T,N}(h_coeff, h_exp)
end

function Base.:*(f::TropicalPuiseuxPolyStatic{T,N}, g::TropicalPuiseuxPolyStatic{T,N}) where {T,N}
    m, n = length(f.exp), length(g.exp)
    result_size = m * n

    result_exp = Vector{SVector{N,T}}(undef, result_size)
    result_coeff = Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(result_coeff, result_size)

    idx = 1
    @inbounds for i in 1:m
        f_exp_i = f.exp[i]
        f_coeff_i = f.coeff[f_exp_i]
        for j in 1:n
            g_exp_j = g.exp[j]
            new_exp = f_exp_i + g_exp_j
            result_exp[idx] = new_exp

            new_coeff = f_coeff_i * g.coeff[g_exp_j]
            if haskey(result_coeff, new_exp)
                result_coeff[new_exp] += new_coeff
            else
                result_coeff[new_exp] = new_coeff
            end
            idx += 1
        end
    end

    # Sort and deduplicate
    sorted_exp = sort(collect(keys(result_coeff)))
    return TropicalPuiseuxPolyStatic{T,N}(result_coeff, sorted_exp)
end

#==============================================================================#
#                    ARITHMETIC: MATRIX IMPLEMENTATION                          #
#==============================================================================#

function Base.:+(f::TropicalPuiseuxPolyMatrix{T}, g::TropicalPuiseuxPolyMatrix{T}) where T
    @assert f.dim == g.dim "Dimensions must match"

    m, n = length(f), length(g)
    dim = f.dim

    # Combine and sort
    combined_exp = Matrix{T}(undef, dim, m + n)
    @inbounds for i in 1:m
        for d in 1:dim
            combined_exp[d, i] = f.exp[d, i]
        end
    end
    @inbounds for j in 1:n
        for d in 1:dim
            combined_exp[d, m + j] = g.exp[d, j]
        end
    end

    combined_coeff = vcat(f.coeff, g.coeff)

    # Sort
    perm = sortperm([combined_exp[:, i] for i in 1:(m+n)])
    sorted_exp = combined_exp[:, perm]
    sorted_coeff = combined_coeff[perm]

    return TropicalPuiseuxPolyMatrix{T}(sorted_exp, sorted_coeff)
end

function Base.:*(f::TropicalPuiseuxPolyMatrix{T}, g::TropicalPuiseuxPolyMatrix{T}) where T
    @assert f.dim == g.dim "Dimensions must match"

    m, n = length(f), length(g)
    dim = f.dim
    result_size = m * n

    result_exp = Matrix{T}(undef, dim, result_size)
    result_coeff = Vector{Oscar.TropicalSemiringElem{typeof(max)}}(undef, result_size)

    idx = 1
    @inbounds for i in 1:m
        f_coeff_i = f.coeff[i]
        for j in 1:n
            for d in 1:dim
                result_exp[d, idx] = f.exp[d, i] + g.exp[d, j]
            end
            result_coeff[idx] = f_coeff_i * g.coeff[j]
            idx += 1
        end
    end

    # Sort
    perm = sortperm([result_exp[:, i] for i in 1:result_size])
    sorted_exp = result_exp[:, perm]
    sorted_coeff = result_coeff[perm]

    return TropicalPuiseuxPolyMatrix{T}(sorted_exp, sorted_coeff)
end

#==============================================================================#
#                         EVALUATION                                            #
#==============================================================================#

function eval_poly(f::TropicalPuiseuxPolyStatic{T,N}, a::Vector) where {T,N}
    ev = zero(a[1])
    for i in eachindex(f)
        exp_i = f.exp[i]
        coeff_i = f.coeff[exp_i]
        term = one(a[1])
        @inbounds for j in 1:N
            term *= a[j]^exp_i[j]
        end
        ev += coeff_i * term
    end
    return ev
end

function eval_poly(f::TropicalPuiseuxPolyMatrix{T}, a::Vector) where T
    ev = zero(a[1])
    dim = f.dim
    for i in 1:length(f)
        coeff_i = f.coeff[i]
        term = one(a[1])
        @inbounds for j in 1:dim
            term *= a[j]^f.exp[j, i]
        end
        ev += coeff_i * term
    end
    return ev
end

#==============================================================================#
#                    SCALAR MULTIPLICATION & EXPONENTIATION                     #
#==============================================================================#

# Scalar multiplication
function Base.:*(c::Oscar.TropicalSemiringElem, f::TropicalPuiseuxPolyStatic{T,N}) where {T,N}
    new_coeff = Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}(
        k => c * v for (k, v) in f.coeff
    )
    return TropicalPuiseuxPolyStatic{T,N}(new_coeff, copy(f.exp))
end

function Base.:*(c::Oscar.TropicalSemiringElem, f::TropicalPuiseuxPolyMatrix{T}) where T
    return TropicalPuiseuxPolyMatrix{T}(copy(f.exp), c .* f.coeff)
end

# Exponentiation by rational
function Base.:^(f::TropicalPuiseuxPolyStatic{T,N}, r::Rational) where {T,N}
    if r == 0
        # Return one polynomial
        R = parent(first(values(f.coeff)))
        one_exp = SVector{N,T}(zeros(T, N))
        return TropicalPuiseuxPolyStatic{T,N}(
            Dict(one_exp => one(R(0))),
            [one_exp]
        )
    end

    new_exp = [SVector{N,T}(T(r * e) for e in exp_vec) for exp_vec in f.exp]
    new_coeff = Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}(
        new_exp[i] => f.coeff[f.exp[i]]^r for i in eachindex(f.exp)
    )
    return TropicalPuiseuxPolyStatic{T,N}(new_coeff, new_exp)
end

function Base.:^(f::TropicalPuiseuxPolyMatrix{T}, r::Rational) where T
    if r == 0
        # Return one polynomial
        R = parent(f.coeff[1])
        return TropicalPuiseuxPolyMatrix{T}(
            zeros(T, f.dim, 1),
            [one(R(0))]
        )
    end

    n = length(f)
    new_exp = Matrix{T}(undef, f.dim, n)
    @inbounds for i in 1:n
        for d in 1:f.dim
            new_exp[d, i] = T(r * f.exp[d, i])
        end
    end
    new_coeff = [c^r for c in f.coeff]
    return TropicalPuiseuxPolyMatrix{T}(new_exp, new_coeff)
end

# Exponentiation by Float64
Base.:^(f::AbstractTropicalPuiseuxPoly, r::Float64) = f^rationalize(r)

# Exponentiation by Int
Base.:^(f::AbstractTropicalPuiseuxPoly, n::Int) = f^Rational(n)

#==============================================================================#
#                    QUICKSUM (FAST SUMMATION)                                  #
#==============================================================================#

"""
    quicksum(F::Vector{<:AbstractTropicalPuiseuxPoly})

Fast summation that defers sorting and deduplication until the end.
Trades accuracy for speed in intermediate steps.
"""
function quicksum(F::Vector{TropicalPuiseuxPolyStatic{T,N}}) where {T,N}
    isempty(F) && throw(ArgumentError("Cannot quicksum empty vector"))

    # Estimate total terms
    total_terms = sum(length(f.exp) for f in F)

    h_coeff = Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(h_coeff, total_terms)
    h_exp = Vector{SVector{N,T}}()
    sizehint!(h_exp, total_terms)

    # Collect all exponents
    @inbounds for f in F
        for exp in f.exp
            push!(h_exp, exp)
        end
    end

    # Sum coefficients
    @inbounds for exp in h_exp
        if !haskey(h_coeff, exp)
            coeff_sum = zero(F[1].coeff[F[1].exp[1]])
            for f in F
                if haskey(f.coeff, exp)
                    coeff_sum += f.coeff[exp]
                end
            end
            h_coeff[exp] = coeff_sum
        end
    end

    return TropicalPuiseuxPolyStatic{T,N}(h_coeff, h_exp)
end

function quicksum(F::Vector{TropicalPuiseuxPolyMatrix{T}}) where T
    isempty(F) && throw(ArgumentError("Cannot quicksum empty vector"))

    dim = F[1].dim
    total_terms = sum(length(f) for f in F)

    # Combine all exponents and coefficients
    combined_exp = Matrix{T}(undef, dim, total_terms)
    combined_coeff = Vector{Oscar.TropicalSemiringElem{typeof(max)}}(undef, total_terms)

    idx = 1
    @inbounds for f in F
        n = length(f)
        for i in 1:n
            for d in 1:dim
                combined_exp[d, idx] = f.exp[d, i]
            end
            combined_coeff[idx] = f.coeff[i]
            idx += 1
        end
    end

    return TropicalPuiseuxPolyMatrix{T}(combined_exp, combined_coeff)
end

#==============================================================================#
#                    MULTIPLICATION WITH QUICKSUM                               #
#==============================================================================#

"""
    mul_with_quicksum(f, g)

Multiplication that defers sorting and deduplication.
Faster but less accurate for intermediate results.
"""
function mul_with_quicksum(f::TropicalPuiseuxPolyStatic{T,N}, g::TropicalPuiseuxPolyStatic{T,N}) where {T,N}
    m, n = length(f.exp), length(g.exp)
    result_size = m * n

    result_exp = Vector{SVector{N,T}}(undef, result_size)
    result_coeff = Dict{SVector{N,T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(result_coeff, result_size)

    idx = 1
    @inbounds for i in 1:m
        f_exp_i = f.exp[i]
        f_coeff_i = f.coeff[f_exp_i]
        for j in 1:n
            g_exp_j = g.exp[j]
            new_exp = f_exp_i + g_exp_j
            new_coeff = f_coeff_i * g.coeff[g_exp_j]

            result_exp[idx] = new_exp
            if haskey(result_coeff, new_exp)
                result_coeff[new_exp] += new_coeff
            else
                result_coeff[new_exp] = new_coeff
            end
            idx += 1
        end
    end

    return TropicalPuiseuxPolyStatic{T,N}(result_coeff, result_exp)
end

function mul_with_quicksum(f::TropicalPuiseuxPolyMatrix{T}, g::TropicalPuiseuxPolyMatrix{T}) where T
    # Same as regular multiplication for matrix version
    return f * g
end

#==============================================================================#
#                    COMPOSITION                                                #
#==============================================================================#

"""
    comp(f::AbstractTropicalPuiseuxPoly, G::Vector{<:AbstractTropicalPuiseuxPoly})

Compose polynomial f with vector of polynomials G.
Computes f(G[1], G[2], ..., G[n]).
"""
function comp(f::TropicalPuiseuxPolyStatic{T,N}, G::Vector{<:AbstractTropicalPuiseuxPoly}) where {T,N}
    @assert length(G) == N "Number of polynomials must match variables"

    # Get a zero polynomial in the output space
    zero_poly = OptimalTropicalPoly(
        [zero(first(values(f.coeff)))],
        [zeros(T, nvars(G[1]))],
        true
    )

    comp = zero_poly

    # Evaluate monomial-wise
    for (exp, coeff) in f.coeff
        term_poly = OptimalTropicalPoly(
            [one(coeff)],
            [zeros(T, nvars(G[1]))],
            true
        )

        for i in 1:N
            term_poly = term_poly * (G[i]^exp[i])
        end
        comp = comp + (coeff * term_poly)
    end

    return comp
end

# Similar implementation for matrix version
function comp(f::TropicalPuiseuxPolyMatrix{T}, G::Vector{<:AbstractTropicalPuiseuxPoly}) where T
    @assert length(G) == f.dim "Number of polynomials must match variables"

    # Get a zero polynomial in the output space
    zero_poly = OptimalTropicalPoly(
        [zero(f.coeff[1])],
        [zeros(T, nvars(G[1]))],
        true
    )

    comp = zero_poly

    # Evaluate monomial-wise
    for i in 1:length(f)
        term_poly = OptimalTropicalPoly(
            [one(f.coeff[i])],
            [zeros(T, nvars(G[1]))],
            true
        )

        for d in 1:f.dim
            term_poly = term_poly * (G[d]^f.exp[d, i])
        end
        comp = comp + (f.coeff[i] * term_poly)
    end

    return comp
end

#==============================================================================#
#                    HELPER CONSTRUCTORS                                        #
#==============================================================================#

"""
    poly_const(n::Int, c::TropicalSemiringElem, ::Type{T}=Float64) where T

Create a constant polynomial in n variables.
"""
function poly_const(n::Int, c::Oscar.TropicalSemiringElem, ::Type{T}=Float64) where T
    return OptimalTropicalPoly([c], [zeros(T, n)], true)
end

"""
    poly_zero(n::Int, R::Oscar.TropicalSemiring, ::Type{T}=Float64) where T

Create the zero polynomial (tropical negative infinity) in n variables.
"""
function poly_zero(n::Int, R, ::Type{T}=Float64) where T
    return poly_const(n, zero(R(0)), T)
end

"""
    poly_one(n::Int, R::Oscar.TropicalSemiring, ::Type{T}=Float64) where T

Create the one polynomial (multiplicative identity, tropical zero) in n variables.
"""
function poly_one(n::Int, R, ::Type{T}=Float64) where T
    return poly_const(n, one(R(0)), T)
end

"""
    poly_monomial(c::TropicalSemiringElem, exp::Vector{T}) where T

Create a monomial polynomial.
"""
function poly_monomial(c::Oscar.TropicalSemiringElem, exp::Vector{T}) where T
    return OptimalTropicalPoly([c], [exp], true)
end

#==============================================================================#
#                    EQUALITY AND STRING REPRESENTATION                         #
#==============================================================================#

function Base.:(==)(f::TropicalPuiseuxPolyStatic{T,N}, g::TropicalPuiseuxPolyStatic{T,N}) where {T,N}
    return f.coeff == g.coeff && f.exp == g.exp
end

function Base.:(==)(f::TropicalPuiseuxPolyMatrix{T}, g::TropicalPuiseuxPolyMatrix{T}) where T
    return f.exp == g.exp && f.coeff == g.coeff
end

function Base.string(f::TropicalPuiseuxPolyStatic{T,N}) where {T,N}
    str = ""
    for (i, exp) in enumerate(f.exp)
        if i > 1
            str *= " + "
        end
        str *= repr(f.coeff[exp])
        for j in 1:N
            str *= " * T_$j^" * repr(exp[j])
        end
    end
    return str
end

function Base.string(f::TropicalPuiseuxPolyMatrix{T}) where T
    str = ""
    for i in 1:length(f)
        if i > 1
            str *= " + "
        end
        str *= repr(f.coeff[i])
        for j in 1:f.dim
            str *= " * T_$j^" * repr(f.exp[j, i])
        end
    end
    return str
end

Base.repr(f::AbstractTropicalPuiseuxPoly) = string(f)

#==============================================================================#
#                    SMART CONSTRUCTOR (AUTO-SELECTS IMPLEMENTATION)           #
#==============================================================================#

"""
    OptimalTropicalPoly(coeff, exp, sorted=false)

Create a tropical Puiseux polynomial using the optimal internal representation
for the given dimension:
- Dimensions 1-5: Uses StaticArrays (~1.4x faster)
- Dimensions >5:  Uses Matrix storage (~2-3x faster at high dims)

The returned type varies based on dimension, but all types conform to the
`AbstractTropicalPuiseuxPoly` interface.

# Example
```julia
R = tropical_semiring(max)

# 2D polynomial - uses StaticArrays internally
f = OptimalTropicalPoly([R(1), R(2)], [[1.0, 2.0], [2.0, 1.0]], false)

# 20D polynomial - uses Matrix storage internally
g = OptimalTropicalPoly([R(1), R(2)], [rand(20) for _ in 1:2], false)

# Both work with the same API
h = f * f  # StaticArrays multiplication
k = g * g  # Matrix multiplication
```
"""
function OptimalTropicalPoly(
    coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
    exp_vecs::Vector{Vector{T}},
    sorted::Bool=false
) where T
    if isempty(exp_vecs)
        throw(ArgumentError("Cannot create polynomial with no terms"))
    end

    dim = length(exp_vecs[1])

    # Choose optimal representation based on dimension
    if dim == 1
        return TropicalPuiseuxPolyStatic{T,1}(coeffs, exp_vecs, sorted)
    elseif dim == 2
        return TropicalPuiseuxPolyStatic{T,2}(coeffs, exp_vecs, sorted)
    elseif dim == 3
        return TropicalPuiseuxPolyStatic{T,3}(coeffs, exp_vecs, sorted)
    elseif dim == 4
        return TropicalPuiseuxPolyStatic{T,4}(coeffs, exp_vecs, sorted)
    elseif dim == 5
        return TropicalPuiseuxPolyStatic{T,5}(coeffs, exp_vecs, sorted)
    else
        return TropicalPuiseuxPolyMatrix{T}(coeffs, exp_vecs, sorted)
    end
end

# Convenience constructor from Dict
function OptimalTropicalPoly(
    coeff_dict::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
    exp_vecs::Vector{Vector{T}},
    sorted::Bool=false
) where T
    coeffs = [coeff_dict[e] for e in exp_vecs]
    return OptimalTropicalPoly(coeffs, exp_vecs, sorted)
end

#==============================================================================#
#                    CONVERSION BETWEEN REPRESENTATIONS                         #
#==============================================================================#

# Note: Conversion functions to/from baseline TropicalPuiseuxPoly are defined
# in a separate extension file that loads when TropicalNN is available.
# For standalone use, the optimal representations work independently.

#==============================================================================#
#                    RATIONAL FUNCTIONS                                         #
#==============================================================================#

"""
    TropicalPuiseuxRationalOpt{P<:AbstractTropicalPuiseuxPoly}

Optimized tropical Puiseux rational function using the same polynomial
representation for both numerator and denominator.
"""
struct TropicalPuiseuxRationalOpt{P<:AbstractTropicalPuiseuxPoly}
    num::P
    den::P

    function TropicalPuiseuxRationalOpt(num::P, den::P) where P<:AbstractTropicalPuiseuxPoly
        new{P}(num, den)
    end
end

# Smart constructor
function OptimalTropicalRational(num_coeffs, num_exp, den_coeffs, den_exp, sorted=false)
    num = OptimalTropicalPoly(num_coeffs, num_exp, sorted)
    den = OptimalTropicalPoly(den_coeffs, den_exp, sorted)
    return TropicalPuiseuxRationalOpt(num, den)
end

# Arithmetic
function Base.:+(f::TropicalPuiseuxRationalOpt{P}, g::TropicalPuiseuxRationalOpt{P}) where P
    num = f.num * g.den + f.den * g.num
    den = f.den * g.den
    return TropicalPuiseuxRationalOpt(num, den)
end

function Base.:*(f::TropicalPuiseuxRationalOpt{P}, g::TropicalPuiseuxRationalOpt{P}) where P
    return TropicalPuiseuxRationalOpt(f.num * g.num, f.den * g.den)
end

function Base.:/(f::TropicalPuiseuxRationalOpt{P}, g::TropicalPuiseuxRationalOpt{P}) where P
    return TropicalPuiseuxRationalOpt(f.num * g.den, f.den * g.num)
end

function eval_rational(f::TropicalPuiseuxRationalOpt, a::Vector)
    return eval_poly(f.num, a) / eval_poly(f.den, a)
end

#==============================================================================#
#                              EXPORTS                                          #
#==============================================================================#

export AbstractTropicalPuiseuxPoly, AbstractTropicalPuiseuxRational
export TropicalPuiseuxPolyStatic, TropicalPuiseuxPolyMatrix
export TropicalPuiseuxRationalOpt
export OptimalTropicalPoly, OptimalTropicalRational
export get_exp, get_coeff, eval_poly, eval_rational
export quicksum, mul_with_quicksum, comp
export poly_const, poly_zero, poly_one, poly_monomial
