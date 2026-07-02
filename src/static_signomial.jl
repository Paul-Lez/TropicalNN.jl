# StaticArrays-based signomial implementation.

"""
    SignomialStatic{T, N}

Tropical Puiseux polynomial whose exponent vectors are stored as
`SVector{N,T}` values.

# Fields
- `coeff`: Coefficients parallel to `exp`
- `exp`: Exponent vectors in sorted lexicographic order
"""
struct SignomialStatic{T, N} <: AbstractSignomial{T}
    coeff::Vector{Oscar.TropicalSemiringElem{typeof(max)}}
    exp::Vector{SVector{N, T}}

    function SignomialStatic{T, N}(
            coeff::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
            exp::Vector{SVector{N, T}}
    ) where {T, N}
        length(coeff) == length(exp) ||
            throw(DimensionMismatch("Coefficient count must match monomial count"))
        @assert issorted(exp) "SignomialStatic exponents must be sorted"
        @assert all(exp[i] != exp[i - 1] for i in 2:length(exp)) "SignomialStatic exponents must be unique"
        new{T, N}(coeff, exp)
    end
end

function _canonicalize_static_terms(
        coeffs::AbstractVector{<:Oscar.TropicalSemiringElem{typeof(max)}},
        exp_static::AbstractVector{SVector{N, T}},
        sorted::Bool = false
) where {T, N}
    length(coeffs) == length(exp_static) ||
        throw(DimensionMismatch("Coefficient count must match exponent count"))

    if isempty(exp_static)
        return Oscar.TropicalSemiringElem{typeof(max)}[], SVector{N, T}[]
    end

    if sorted
        coeff_work = collect(coeffs)
        exp_work = collect(exp_static)
    else
        perm = sortperm(exp_static)
        coeff_work = coeffs[perm]
        exp_work = exp_static[perm]
    end

    canonical_coeff = Oscar.TropicalSemiringElem{typeof(max)}[]
    canonical_exp = SVector{N, T}[]
    sizehint!(canonical_coeff, length(exp_work))
    sizehint!(canonical_exp, length(exp_work))

    @inbounds for i in eachindex(exp_work)
        if !isempty(canonical_exp) && exp_work[i] == canonical_exp[end]
            canonical_coeff[end] += coeff_work[i]
        else
            push!(canonical_exp, exp_work[i])
            push!(canonical_coeff, coeff_work[i])
        end
    end

    return canonical_coeff, canonical_exp
end

function SignomialStatic{T, N}(
        coeff_dict::Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp_static::Vector{SVector{N, T}}
) where {T, N}
    coeffs = Oscar.TropicalSemiringElem{typeof(max)}[coeff_dict[e] for e in exp_static]
    coeffs, exp_static = _canonicalize_static_terms(coeffs, exp_static, issorted(exp_static))
    return SignomialStatic{T, N}(coeffs, exp_static)
end

# Constructor from vector-of-vectors
function SignomialStatic{T, N}(
        coeff_dict::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T, N}
    exp_static = [SVector{N, T}(e) for e in exp_vecs]
    coeffs = Oscar.TropicalSemiringElem{typeof(max)}[coeff_dict[e] for e in exp_vecs]
    coeffs, exp_static = _canonicalize_static_terms(coeffs, exp_static, sorted)
    return SignomialStatic{T, N}(coeffs, exp_static)
end

# Constructor from coefficient vector and exponent vector
function SignomialStatic{T, N}(
        coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T, N}
    exp_static = [SVector{N, T}(e) for e in exp_vecs]
    coeffs, exp_static = _canonicalize_static_terms(coeffs, exp_static, sorted)
    return SignomialStatic{T, N}(coeffs, exp_static)
end

Oscar.nvars(f::SignomialStatic{T, N}) where {T, N} = N
Base.length(f::SignomialStatic) = length(f.exp)

function get_exp(f::SignomialStatic{T, N}, i::Int) where {T, N}
    return f.exp[i]
end

function get_coeff(f::SignomialStatic, i::Int)
    return f.coeff[i]
end

function Base.:+(f::SignomialStatic{T, N}, g::SignomialStatic{T, N}) where {T, N}
    @assert issorted(f.exp) "SignomialStatic exponents must be sorted before addition"
    @assert issorted(g.exp) "SignomialStatic exponents must be sorted before addition"

    lf, lg = length(f.exp), length(g.exp)

    h_coeff = Oscar.TropicalSemiringElem{typeof(max)}[]
    sizehint!(h_coeff, lf + lg)
    h_exp = Vector{SVector{N, T}}()
    sizehint!(h_exp, lf + lg)

    trop_zero = if lf > 0
        zero(f.coeff[1])
    elseif lg > 0
        zero(g.coeff[1])
    else
        _tropical_zero(f)
    end

    # Merge sorted lists
    i, j = 1, 1
    @inbounds while i <= lf && j <= lg
        f_exp, g_exp = f.exp[i], g.exp[j]
        if f_exp < g_exp
            c = f.coeff[i]
            if c != trop_zero
                push!(h_exp, f_exp)
                push!(h_coeff, c)
            end
            i += 1
        elseif g_exp < f_exp
            c = g.coeff[j]
            if c != trop_zero
                push!(h_exp, g_exp)
                push!(h_coeff, c)
            end
            j += 1
        else  # equal
            c = f.coeff[i] + g.coeff[j]
            if c != trop_zero
                push!(h_exp, f_exp)
                push!(h_coeff, c)
            end
            i += 1
            j += 1
        end
    end

    # Remaining from f
    @inbounds while i <= lf
        f_exp = f.exp[i]
        c = f.coeff[i]
        if c != trop_zero
            push!(h_exp, f_exp)
            push!(h_coeff, c)
        end
        i += 1
    end

    # Remaining from g
    @inbounds while j <= lg
        g_exp = g.exp[j]
        c = g.coeff[j]
        if c != trop_zero
            push!(h_exp, g_exp)
            push!(h_coeff, c)
        end
        j += 1
    end

    return SignomialStatic{T, N}(h_coeff, h_exp)
end

function Base.:*(f::SignomialStatic{T, N}, g::SignomialStatic{T, N}) where {T, N}
    m, n = length(f.exp), length(g.exp)
    result_size = m * n

    result_exp = Vector{SVector{N, T}}(undef, result_size)
    result_coeff = Vector{Oscar.TropicalSemiringElem{typeof(max)}}(undef, result_size)

    idx = 1
    @inbounds for i in 1:m
        f_exp_i = f.exp[i]
        f_coeff_i = f.coeff[i]
        for j in 1:n
            g_exp_j = g.exp[j]
            result_exp[idx] = f_exp_i + g_exp_j
            result_coeff[idx] = f_coeff_i * g.coeff[j]
            idx += 1
        end
    end

    result_coeff, result_exp = _canonicalize_static_terms(result_coeff, result_exp, false)
    return SignomialStatic{T, N}(result_coeff, result_exp)
end

function eval_poly(f::SignomialStatic{T, N}, a::Vector) where {T, N}
    ev = zero(a[1])
    for i in Base.eachindex(f)
        exp_i = f.exp[i]
        coeff_i = f.coeff[i]
        term = one(a[1])
        @inbounds for j in 1:N
            term *= a[j]^exp_i[j]
        end
        ev += coeff_i * term
    end
    return ev
end

function Base.:*(c::Oscar.TropicalSemiringElem, f::SignomialStatic{T, N}) where {T, N}
    new_coeff = Oscar.TropicalSemiringElem{typeof(max)}[c * v for v in f.coeff]
    return SignomialStatic{T, N}(new_coeff, copy(f.exp))
end

function Base.:^(f::SignomialStatic{T, N}, r::Base.Rational) where {T, N}
    if r == 0
        # Return one polynomial
        one_exp = SVector{N, T}(zeros(T, N))
        return SignomialStatic{T, N}(
            Oscar.TropicalSemiringElem{typeof(max)}[_tropical_one(f)],
            [one_exp]
        )
    end

    new_exp = [SVector{N, T}(T(r * e) for e in exp_vec) for exp_vec in f.exp]
    new_coeff = Oscar.TropicalSemiringElem{typeof(max)}[f.coeff[i]^r
                                                        for i in Base.eachindex(f.exp)]
    new_coeff, new_exp = _canonicalize_static_terms(new_coeff, new_exp, false)
    return SignomialStatic{T, N}(new_coeff, new_exp)
end

function quicksum(F::Vector{SignomialStatic{T, N}}) where {T, N}
    isempty(F) && throw(ArgumentError("Cannot quicksum empty vector"))

    # Estimate total terms
    total_terms = sum(length(f.exp) for f in F)

    h_coeff = Vector{Oscar.TropicalSemiringElem{typeof(max)}}(undef, total_terms)
    h_exp = Vector{SVector{N, T}}(undef, total_terms)

    idx = 1
    @inbounds for f in F
        for i in eachindex(f.exp)
            h_exp[idx] = f.exp[i]
            h_coeff[idx] = f.coeff[i]
            idx += 1
        end
    end

    h_coeff, h_exp = _canonicalize_static_terms(h_coeff, h_exp, false)
    return SignomialStatic{T, N}(h_coeff, h_exp)
end

function comp(f::SignomialStatic{T, N}, G::Vector{<:AbstractSignomial}) where {T, N}
    @assert length(G) == N "Number of polynomials must match variables"

    # Get a zero polynomial in the output space
    zero_poly = Signomial(
        [_tropical_zero(f)],
        [zeros(T, nvars(G[1]))],
        true
    )

    result = zero_poly

    # Evaluate monomial-wise
    for i in eachindex(f.exp)
        exp = f.exp[i]
        coeff = f.coeff[i]
        term_poly = Signomial(
            [one(coeff)],
            [zeros(T, nvars(G[1]))],
            true
        )

        for i in 1:N
            term_poly = term_poly * (G[i]^exp[i])
        end
        result = result + (coeff * term_poly)
    end

    return result
end

function Base.:(==)(f::SignomialStatic{T, N}, g::SignomialStatic{T, N}) where {T, N}
    return f.coeff == g.coeff && f.exp == g.exp
end

function Base.string(f::SignomialStatic{T, N}) where {T, N}
    str = ""
    for (i, exp) in enumerate(f.exp)
        if i > 1
            str *= " + "
        end
        str *= repr(f.coeff[i])
        for j in 1:N
            str *= " * T_$j^" * repr(exp[j])
        end
    end
    return str
end

function get_coeff_by_exp(f::SignomialStatic{T, N}, e) where {T, N}
    exp = SVector{N, T}(e)
    idx = searchsortedfirst(f.exp, exp)
    if idx <= length(f.exp) && f.exp[idx] == exp
        return f.coeff[idx]
    end
    throw(KeyError(e))
end

function exponents(f::SignomialStatic)
    return copy(f.exp)
end

function coefficients(f::SignomialStatic)
    return copy(f.coeff)
end
