# StaticArrays-based signomial implementation.

"""
    SignomialStatic{T, N}

Tropical Puiseux polynomial whose exponent vectors are stored as
`SVector{N,T}` values.

# Fields
- `coeff`: Map from exponent vectors to tropical coefficients
- `exp`: Exponent vectors in iteration order
"""
struct SignomialStatic{T, N} <: AbstractSignomial{T}
    coeff::Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}
    exp::Vector{SVector{N, T}}

    function SignomialStatic{T, N}(
            coeff::Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}},
            exp::Vector{SVector{N, T}}
    ) where {T, N}
        new{T, N}(coeff, exp)
    end
end

# Constructor from vector-of-vectors
function SignomialStatic{T, N}(
        coeff_dict::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T, N}
    # Convert to static vectors
    exp_static = [SVector{N, T}(e) for e in exp_vecs]
    coeff_static = Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    for (k, v) in coeff_dict
        coeff_static[SVector{N, T}(k)] = v
    end

    # Sort if needed
    if !sorted
        sort!(exp_static)
    end

    return SignomialStatic{T, N}(coeff_static, exp_static)
end

# Constructor from coefficient vector and exponent vector
function SignomialStatic{T, N}(
        coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T, N}
    exp_static = [SVector{N, T}(e) for e in exp_vecs]

    if !sorted
        perm = sortperm(exp_static)
        exp_static = exp_static[perm]
        coeffs = coeffs[perm]
    end

    coeff_static = Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}(
        exp_static[i] => coeffs[i] for i in Base.eachindex(coeffs)
    )

    return SignomialStatic{T, N}(coeff_static, exp_static)
end

Oscar.nvars(f::SignomialStatic{T, N}) where {T, N} = N
Base.length(f::SignomialStatic) = length(f.exp)

function get_exp(f::SignomialStatic{T, N}, i::Int) where {T, N}
    return f.exp[i]
end

function get_coeff(f::SignomialStatic, i::Int)
    return f.coeff[f.exp[i]]
end

function Base.:+(f::SignomialStatic{T, N}, g::SignomialStatic{T, N}) where {T, N}
    lf, lg = length(f.exp), length(g.exp)

    h_coeff = Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(h_coeff, lf + lg)
    h_exp = Vector{SVector{N, T}}()
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

    return SignomialStatic{T, N}(h_coeff, h_exp)
end

function Base.:*(f::SignomialStatic{T, N}, g::SignomialStatic{T, N}) where {T, N}
    m, n = length(f.exp), length(g.exp)
    result_size = m * n

    result_exp = Vector{SVector{N, T}}(undef, result_size)
    result_coeff = Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}()
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
    return SignomialStatic{T, N}(result_coeff, sorted_exp)
end

function eval_poly(f::SignomialStatic{T, N}, a::Vector) where {T, N}
    ev = zero(a[1])
    for i in Base.eachindex(f)
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

function Base.:*(c::Oscar.TropicalSemiringElem, f::SignomialStatic{T, N}) where {T, N}
    new_coeff = Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}(
        k => c * v for (k, v) in f.coeff
    )
    return SignomialStatic{T, N}(new_coeff, copy(f.exp))
end

function Base.:^(f::SignomialStatic{T, N}, r::Base.Rational) where {T, N}
    if r == 0
        # Return one polynomial
        R = parent(first(values(f.coeff)))
        one_exp = SVector{N, T}(zeros(T, N))
        return SignomialStatic{T, N}(
            Dict(one_exp => one(R(0))),
            [one_exp]
        )
    end

    new_exp = [SVector{N, T}(T(r * e) for e in exp_vec) for exp_vec in f.exp]
    new_coeff = Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}(
        new_exp[i] => f.coeff[f.exp[i]]^r for i in Base.eachindex(f.exp)
    )
    return SignomialStatic{T, N}(new_coeff, new_exp)
end

function quicksum(F::Vector{SignomialStatic{T, N}}) where {T, N}
    isempty(F) && throw(ArgumentError("Cannot quicksum empty vector"))

    # Estimate total terms
    total_terms = sum(length(f.exp) for f in F)

    h_coeff = Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(h_coeff, total_terms)
    h_exp = Vector{SVector{N, T}}()
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

    return SignomialStatic{T, N}(h_coeff, h_exp)
end

function mul_with_quicksum(f::SignomialStatic{T, N}, g::SignomialStatic{T, N}) where {T, N}
    m, n = length(f.exp), length(g.exp)
    result_size = m * n

    result_exp = Vector{SVector{N, T}}(undef, result_size)
    result_coeff = Dict{SVector{N, T}, Oscar.TropicalSemiringElem{typeof(max)}}()
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

    return SignomialStatic{T, N}(result_coeff, result_exp)
end

function comp(f::SignomialStatic{T, N}, G::Vector{<:AbstractSignomial}) where {T, N}
    @assert length(G) == N "Number of polynomials must match variables"

    # Get a zero polynomial in the output space
    zero_poly = OptimalTropicalPoly(
        [zero(first(values(f.coeff)))],
        [zeros(T, nvars(G[1]))],
        true
    )

    result = zero_poly

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
        str *= repr(f.coeff[exp])
        for j in 1:N
            str *= " * T_$j^" * repr(exp[j])
        end
    end
    return str
end

function get_coeff_by_exp(f::SignomialStatic{T, N}, e) where {T, N}
    return f.coeff[SVector{N, T}(e)]
end

function exponents(f::SignomialStatic)
    return f.exp
end

function coefficients(f::SignomialStatic)
    return [f.coeff[e] for e in f.exp]
end
