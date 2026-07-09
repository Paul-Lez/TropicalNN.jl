# Matrix-based signomial implementation and shared signomial API.
#
# Signomials are stored as a coefficient vector together with an exponent
# matrix whose columns are exponent vectors. This is the only concrete
# representation used by the public Signomial constructor.

const _TROPICAL_COEFF = Oscar.TropicalSemiringElem{typeof(max)}

function _coefficient_parent(f)
    if length(f) == 0
        return Oscar.tropical_semiring(max)
    end
    return parent(get_coeff(f, 1))
end

_tropical_zero(f) = zero(_coefficient_parent(f)(0))
_tropical_one(f) = one(_coefficient_parent(f)(0))

"""
    _canonize_terms(coeffs, exp_vecs)

Return `(coeffs, exp_vecs)` in canonical signomial form.

This validates that each coefficient has a matching exponent vector and that
all exponent vectors have the same dimension. Duplicate exponent vectors are
merged by tropical addition of their coefficients, and the returned exponent
vectors are sorted lexicographically.
"""
function _canonize_terms(
        coeffs::AbstractVector{<:_TROPICAL_COEFF},
        exp_vecs::AbstractVector{<:AbstractVector{T}}
) where {T}
    length(coeffs) == length(exp_vecs) ||
        throw(DimensionMismatch("Coefficient count must match exponent count"))

    if isempty(exp_vecs)
        return _TROPICAL_COEFF[], Vector{Vector{T}}()
    end

    dim = length(exp_vecs[begin])
    coeff_by_exp = Dict{Tuple, _TROPICAL_COEFF}()
    exp_by_key = Dict{Tuple, Vector{T}}()

    for (c, exp_vec) in zip(coeffs, exp_vecs)
        length(exp_vec) == dim ||
            throw(DimensionMismatch("All exponent vectors must have length $dim"))
        exp = Vector{T}(exp_vec)
        key = Tuple(exp)
        if haskey(coeff_by_exp, key)
            coeff_by_exp[key] += c
        else
            coeff_by_exp[key] = c
            exp_by_key[key] = exp
        end
    end

    exps = collect(values(exp_by_key))
    sort!(exps)
    return _TROPICAL_COEFF[coeff_by_exp[Tuple(exp)] for exp in exps], exps
end

#==============================================================================#
#                    CONCRETE REPRESENTATION                                    #
#==============================================================================#

# Allocation-free lexicographic column comparison used to validate constructor
# fast paths that skip re-canonicalizing already sorted, unique exponent columns.
function _matrix_col_lexless(exp::Matrix, i::Int, j::Int)
    @inbounds for d in axes(exp, 1)
        exp[d, i] < exp[d, j] && return true
        exp[d, j] < exp[d, i] && return false
    end
    return false
end

"""
    Signomial{T}

Tropical Puiseux polynomial whose exponent vectors are stored as columns of a
matrix.

# Fields
- `exp::Matrix{T}`: Exponent matrix (dim x n_monomials), each column is one exponent
- `coeff::Vector{TropicalSemiringElem}`: Coefficients parallel to exp columns
- `dim::Int`: Dimension (number of variables)
"""
struct Signomial{T}
    exp::Matrix{T}
    coeff::Vector{Oscar.TropicalSemiringElem{typeof(max)}}
    dim::Int

    function Signomial{T}(
            exp::Matrix{T},
            coeff::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
            sorted_unique::Bool = false
    ) where {T}
        dim, n_monomials = size(exp)
        @assert length(coeff) == n_monomials "Coefficient count must match monomial count"
        if n_monomials <= 1
            return new{T}(exp, coeff, dim)
        end
        if sorted_unique
            @assert all(_matrix_col_lexless(exp, i - 1, i)
            for i in 2:n_monomials) "Signomial exponents must be sorted and unique"
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
function Signomial{T}(
        coeff_dict::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T}
    if isempty(exp_vecs)
        return Signomial{T}(Matrix{T}(undef, 0, 0), eltype(values(coeff_dict))[])
    end
    coeffs = Oscar.TropicalSemiringElem{typeof(max)}[coeff_dict[e] for e in exp_vecs]
    return Signomial{T}(coeffs, exp_vecs, sorted)
end

# Constructor from coefficient vector and exponent vector
function Signomial{T}(
        coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T}
    if isempty(exp_vecs)
        return Signomial{T}(Matrix{T}(undef, 0, 0), coeffs)
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

    return Signomial{T}(exp_matrix, coeffs)
end

Oscar.nvars(f::Signomial) = f.dim
Base.length(f::Signomial) = size(f.exp, 2)

function get_exp(f::Signomial, i::Int)
    return @view f.exp[:, i]
end

function get_coeff(f::Signomial, i::Int)
    return f.coeff[i]
end

function _remove_zero_matrix_terms(f::Signomial{T}) where {T}
    length(f) == 0 && return f

    tropical_zero = zero(f.coeff[1])
    keep_count = count(!=(tropical_zero), f.coeff)
    keep_count == length(f) && return f

    new_exp = Matrix{T}(undef, f.dim, keep_count)
    new_coeff = Vector{Oscar.TropicalSemiringElem{typeof(max)}}(undef, keep_count)

    idx = 1
    @inbounds for i in Base.eachindex(f)
        c = f.coeff[i]
        if c != tropical_zero
            for d in 1:f.dim
                new_exp[d, idx] = f.exp[d, i]
            end
            new_coeff[idx] = c
            idx += 1
        end
    end

    return Signomial{T}(new_exp, new_coeff, true)
end

function Base.:+(f::Signomial{T}, g::Signomial{T}) where {T}
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

    return _remove_zero_matrix_terms(Signomial{T}(combined_exp, combined_coeff))
end

function Base.:*(f::Signomial{T}, g::Signomial{T}) where {T}
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

    return Signomial{T}(result_exp, result_coeff)
end

function eval_poly(f::Signomial{T}, a::Vector) where {T}
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

function Base.:*(c::Oscar.TropicalSemiringElem, f::Signomial{T}) where {T}
    return Signomial{T}(copy(f.exp), c .* f.coeff)
end

function Base.:^(f::Signomial{T}, r::Base.Rational) where {T}
    if r == 0
        # Return one polynomial
        return Signomial{T}(
            zeros(T, f.dim, 1),
            [_tropical_one(f)]
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
    return Signomial{T}(new_exp, new_coeff)
end

function quicksum(F::Vector{Signomial{T}}) where {T}
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

    return Signomial{T}(combined_exp, combined_coeff)
end

function quicksum(F::Vector{<:Signomial})
    isempty(F) && throw(ArgumentError("Cannot quicksum empty vector"))
    result = F[1]
    for i in 2:length(F)
        result = result + F[i]
    end
    return result
end

function comp(f::Signomial{T}, G::Vector{<:Signomial}) where {T}
    @assert length(G) == f.dim "Number of polynomials must match variables"

    # Get a zero polynomial in the output space
    zero_poly = Signomial(
        [_tropical_zero(f)],
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

function Base.:(==)(f::Signomial{T}, g::Signomial{T}) where {T}
    return f.exp == g.exp && f.coeff == g.coeff
end

function Base.string(f::Signomial{T}) where {T}
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

function get_coeff_by_exp(f::Signomial{T}, e) where {T}
    exp = Vector{T}(e)
    i = findfirst(x -> Vector{T}(x) == exp, eachcol(f.exp))
    if i !== nothing
        return f.coeff[i]
    else
        throw(KeyError(e))
    end
end

function exponents(f::Signomial{T}) where {T}
    return [Vector{T}(f.exp[:, i]) for i in 1:length(f)]
end

function coefficients(f::Signomial)
    return copy(f.coeff)
end

#==============================================================================#
#                         COMMON INTERFACE METHODS                              #
#==============================================================================#

Base.eachindex(f::Signomial) = Base.OneTo(length(f))

# Exponentiation by Float64
Base.:^(f::Signomial, r::Float64) = f^rationalize(r)

# Exponentiation by Int
Base.:^(f::Signomial, n::Int) = f^Base.Rational(n)

#==============================================================================#
#                    STRING REPRESENTATION                                      #
#==============================================================================#

Base.repr(f::Signomial) = string(f)

#==============================================================================#
#                    USER-FACING CONSTRUCTOR                                    #
#==============================================================================#

# Public constructors
function Signomial(
        coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{<:AbstractVector{T}};
        sorted::Bool = false
) where {T}
    return Signomial{T}(coeffs, [Vector{T}(e) for e in exp_vecs], sorted)
end

function Signomial(
        coeff_dict::Dict{<:AbstractVector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{<:AbstractVector{T}};
        sorted::Bool = false
) where {T}
    plain_exps = [Vector{T}(e) for e in exp_vecs]
    plain_dict = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}(
        Vector{T}(k) => v for (k, v) in coeff_dict
    )
    return Signomial{T}(plain_dict, plain_exps, sorted)
end

# Convenience constructor: plain Real coefficients -> wrapped in tropical semiring
function Signomial(
        coeffs::Vector{<:Real},
        exp_vecs::Vector{<:AbstractVector{T}};
        sorted::Bool = false
) where {T}
    R = Oscar.tropical_semiring(max)
    return Signomial{T}(
        Oscar.TropicalSemiringElem{typeof(max)}[R(c) for c in coeffs],
        [Vector{T}(e) for e in exp_vecs], sorted)
end

#==============================================================================#
#                    ADDITIONAL ACCESSOR FUNCTIONS                              #
#==============================================================================#

"""
    monomial_pairs(f::Signomial)

Return an iterable of `(exp, coeff)` pairs in lexicographic exponent order.
"""
function monomial_pairs(f::Signomial)
    return zip(exponents(f), coefficients(f))
end

#==============================================================================#
#                    FACTORY CONSTRUCTORS                                       #
#==============================================================================#

"""
    Signomial_const(n, c, f::Signomial{T})

Construct the constant `c` as a signomial in `n` variables.
`f` is used only to infer the exponent numeric type `T`.
"""
function Signomial_const(n, c, f::Signomial{T}) where {T}
    return Signomial([c], [zeros(T, n)]; sorted = true)
end

"""
    Signomial_zero(n, f::Signomial)

Construct the tropical-zero (additive identity, -infinity) signomial in `n` variables.
`f` is used only to infer the exponent type.
"""
function Signomial_zero(n::Int, f::Signomial)
    return Signomial_const(n, _tropical_zero(f), f)
end

"""
    Signomial_one(n, f::Signomial)

Construct the tropical-one (multiplicative identity, value 0) signomial in `n` variables.
`f` is used only to infer the exponent type.
"""
function Signomial_one(n::Int, f::Signomial)
    return Signomial_const(n, _tropical_one(f), f)
end

"""
    SignomialMonomial(c, exp::Vector{T})

Construct a single-monomial signomial with coefficient `c` and exponent vector `exp`.
"""
function SignomialMonomial(c, exp::Vector{T}) where {T}
    return Signomial([c], [exp]; sorted = true)
end

#==============================================================================#
#                    EVALUATE ALIASES                                           #
#==============================================================================#

"""
    evaluate(f::Signomial, a::Vector)

Evaluate the tropical polynomial `f` at the point `a`.
Alias for `eval_poly`.
"""
function evaluate(f::Signomial, a::Vector)
    return eval_poly(f, _coerce_evaluation_point(f, a))
end

_lift_evaluation_scalar(_, x::Oscar.TropicalSemiringElem) = x

function _lift_evaluation_scalar(R, x::Real)
    return R(x isa AbstractFloat ? rationalize(x) : x)
end

function _coerce_evaluation_point(f::Signomial, a::Vector)
    all(x -> x isa Oscar.TropicalSemiringElem, a) && return a
    R = _coefficient_parent(f)
    return [_lift_evaluation_scalar(R, x) for x in a]
end

# Callable syntax: f(x) as sugar for evaluate(f, x)
(f::Signomial)(x::Vector) = evaluate(f, x)

#==============================================================================#
#                    DEDUP AND MONOMIAL COUNT                                   #
#==============================================================================#

"""
    dedup_monomials(f)

Remove all monomials with tropical-zero coefficient from `f`.
"""
function dedup_monomials(f::Signomial{T}) where {T}
    length(f) == 0 && return f
    keep_count = monomial_count(f)
    keep_count == length(f) && return f

    tropical_zero = zero(f.coeff[1])
    new_coeffs = Vector{Oscar.TropicalSemiringElem{typeof(max)}}(undef, keep_count)
    new_exp = Matrix{T}(undef, f.dim, keep_count)

    idx = 1
    @inbounds for i in Base.eachindex(f)
        c = f.coeff[i]
        if c != tropical_zero
            new_coeffs[idx] = c
            for d in 1:f.dim
                new_exp[d, idx] = f.exp[d, i]
            end
            idx += 1
        end
    end
    return Signomial{T}(new_exp, new_coeffs, true)
end

"""
    monomial_count(f::Signomial)

Return the number of monomials in `f`.
"""
function monomial_count(f::Signomial)
    length(f) == 0 && return 0
    tropical_zero = zero(get_coeff(f, 1))
    n = 0
    for i in Base.eachindex(f)
        if get_coeff(f, i) != tropical_zero
            n += 1
        end
    end
    return n
end

#==============================================================================#
#                    EXPONENTIATION EXTENSIONS                                  #
#==============================================================================#

# TropicalSemiringElem ^ TropicalSemiringElem
function Base.:^(
        a::Oscar.TropicalSemiringElem{typeof(max)},
        b::Oscar.TropicalSemiringElem{typeof(max)}
)
    R = tropical_semiring(max)
    return R(Rational(a) * Rational(b))
end

# TropicalSemiringElem ^ Rational
function Base.:^(a::Oscar.TropicalSemiringElem{typeof(max)}, b::Rational{T}) where {
        T <: Integer}
    R = tropical_semiring(max)
    return R(Rational(a) * b)
end

# TropicalSemiringElem ^ Float64
function Base.:^(a::Oscar.TropicalSemiringElem{typeof(max)}, b::Float64)
    R = tropical_semiring(max)
    return R(rationalize(Float64(Rational(a)) * b))
end

#==============================================================================#
#                    PRETTY PRINTING                                            #
#==============================================================================#

const _SUBSCRIPT_DIGITS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']

function _subscript(n::Integer)
    return join(_SUBSCRIPT_DIGITS[d + 1] for d in reverse(digits(n)))
end

_exp_str(α::AbstractFloat) = isinteger(α) ? "$(Int(α))" : "$(α)"
_exp_str(α::Rational) = denominator(α) == 1 ? "$(numerator(α))" : "$(α)"
_exp_str(α::Integer) = "$(α)"
_exp_str(α) = "$(α)"

_coeff_str(c::AbstractFloat) = isinteger(c) ? "$(Int(c))" : "$(c)"
_coeff_str(c::Rational) = denominator(c) == 1 ? "$(numerator(c))" : "$(c)"
_coeff_str(c::Integer) = "$(c)"
_coeff_str(c) = "$(c)"

function _monomial_str(coeff::Oscar.TropicalSemiringElem, exp)
    c = Oscar.data(coeff)
    terms = String[]
    for (i, α) in enumerate(exp)
        α == 0 && continue
        vname = "x$(_subscript(i))"
        if α == 1
            push!(terms, vname)
        elseif α == -1
            push!(terms, "-" * vname)
        else
            push!(terms, "$(_exp_str(α))$(vname)")
        end
    end
    if isempty(terms)
        return _coeff_str(c)
    end
    c != 0 && pushfirst!(terms, _coeff_str(c))
    return join(terms, " + ")
end

function Base.show(io::IO, f::Signomial)
    if length(f) == 0
        print(io, "max()")
        return
    end
    strs = [_monomial_str(get_coeff(f, i), get_exp(f, i)) for i in Base.eachindex(f)]
    print(io, "max(", join(strs, ", "), ")")
end
