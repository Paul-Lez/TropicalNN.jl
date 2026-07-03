# Matrix-based signomial implementation.

"""
    SignomialMatrix{T}

Tropical Puiseux polynomial whose exponent vectors are stored as columns of a
matrix.

# Fields
- `exp::Matrix{T}`: Exponent matrix (dim x n_monomials), each column is one exponent
- `coeff::Vector{TropicalSemiringElem}`: Coefficients parallel to exp columns
- `dim::Int`: Dimension (number of variables)
"""
struct SignomialMatrix{T} <: AbstractSignomial{T}
    exp::Matrix{T}
    coeff::Vector{Oscar.TropicalSemiringElem{typeof(max)}}
    dim::Int

    function SignomialMatrix{T}(
            exp::Matrix{T},
            coeff::Vector{Oscar.TropicalSemiringElem{typeof(max)}}
    ) where {T}
        dim, n_monomials = size(exp)
        @assert length(coeff) == n_monomials "Coefficient count must match monomial count"
        if n_monomials <= 1
            return new{T}(exp, coeff, dim)
        end

        exp_vecs = [Vector{T}(exp[:, i]) for i in 1:n_monomials]
        canonical_coeff, canonical_exp = _canonize_terms(coeff, exp_vecs)
        canonical_matrix = Matrix{T}(undef, dim, length(canonical_exp))
        @inbounds for i in eachindex(canonical_exp)
            for d in 1:dim
                canonical_matrix[d, i] = canonical_exp[i][d]
            end
        end
        new{T}(canonical_matrix, canonical_coeff, dim)
    end
end

# Constructor from vector-of-vectors
function SignomialMatrix{T}(
        coeff_dict::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T}
    if isempty(exp_vecs)
        return SignomialMatrix{T}(Matrix{T}(undef, 0, 0), eltype(values(coeff_dict))[])
    end
    coeffs = Oscar.TropicalSemiringElem{typeof(max)}[coeff_dict[e] for e in exp_vecs]
    return SignomialMatrix{T}(coeffs, exp_vecs, sorted)
end

# Constructor from coefficient vector and exponent vector
function SignomialMatrix{T}(
        coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T}
    if isempty(exp_vecs)
        return SignomialMatrix{T}(Matrix{T}(undef, 0, 0), coeffs)
    end

    coeffs, exp_vecs = _canonize_terms(coeffs, exp_vecs)
    dim = length(exp_vecs[1])
    n_monomials = length(exp_vecs)

    exp_matrix = Matrix{T}(undef, dim, n_monomials)
    @inbounds for i in 1:n_monomials
        for d in 1:dim
            exp_matrix[d, i] = exp_vecs[i][d]
        end
    end

    return SignomialMatrix{T}(exp_matrix, coeffs)
end

Oscar.nvars(f::SignomialMatrix) = f.dim
Base.length(f::SignomialMatrix) = size(f.exp, 2)

function get_exp(f::SignomialMatrix, i::Int)
    return @view f.exp[:, i]
end

function get_coeff(f::SignomialMatrix, i::Int)
    return f.coeff[i]
end

function Base.:+(f::SignomialMatrix{T}, g::SignomialMatrix{T}) where {T}
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

    return SignomialMatrix{T}(combined_exp, combined_coeff)
end

function Base.:*(f::SignomialMatrix{T}, g::SignomialMatrix{T}) where {T}
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

    return SignomialMatrix{T}(result_exp, result_coeff)
end

function eval_poly(f::SignomialMatrix{T}, a::Vector) where {T}
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

function Base.:*(c::Oscar.TropicalSemiringElem, f::SignomialMatrix{T}) where {T}
    return SignomialMatrix{T}(copy(f.exp), c .* f.coeff)
end

function Base.:^(f::SignomialMatrix{T}, r::Base.Rational) where {T}
    if r == 0
        # Return one polynomial
        R = parent(f.coeff[1])
        return SignomialMatrix{T}(
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
    new_coeff = Oscar.TropicalSemiringElem{typeof(max)}[c^r for c in f.coeff]
    return SignomialMatrix{T}(new_exp, new_coeff)
end

function quicksum(F::Vector{SignomialMatrix{T}}) where {T}
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

    return SignomialMatrix{T}(combined_exp, combined_coeff)
end

function mul_with_quicksum(f::SignomialMatrix{T}, g::SignomialMatrix{T}) where {T}
    # Same as regular multiplication for matrix version
    return f * g
end

function comp(f::SignomialMatrix{T}, G::Vector{<:AbstractSignomial}) where {T}
    @assert length(G) == f.dim "Number of polynomials must match variables"

    # Get a zero polynomial in the output space
    zero_poly = Signomial(
        [zero(f.coeff[1])],
        [zeros(T, nvars(G[1]))],
        true
    )

    result = zero_poly

    # Evaluate monomial-wise
    for i in 1:length(f)
        term_poly = Signomial(
            [one(f.coeff[i])],
            [zeros(T, nvars(G[1]))],
            true
        )

        for d in 1:f.dim
            term_poly = term_poly * (G[d]^f.exp[d, i])
        end
        result = result + (f.coeff[i] * term_poly)
    end

    return result
end

function Base.:(==)(f::SignomialMatrix{T}, g::SignomialMatrix{T}) where {T}
    return f.exp == g.exp && f.coeff == g.coeff
end

function Base.string(f::SignomialMatrix{T}) where {T}
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

function get_coeff_by_exp(f::SignomialMatrix{T}, e) where {T}
    exp = Vector{T}(e)
    i = findfirst(x -> Vector{T}(x) == exp, eachcol(f.exp))
    if i !== nothing
        return f.coeff[i]
    else
        throw(KeyError(e))
    end
end

function exponents(f::SignomialMatrix{T}) where {T}
    return [Vector{T}(f.exp[:, i]) for i in 1:length(f)]
end

function coefficients(f::SignomialMatrix)
    return f.coeff
end
