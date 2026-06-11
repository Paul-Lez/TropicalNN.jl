# Unified interface for tropical Puiseux polynomials
#
# This module provides a unified API for tropical polynomials. The constructor
# stores low-dimensional exponents in StaticArrays and larger exponent sets in
# matrix form.
#
# The user-facing API is dimension-agnostic - the goal here is that implementation details are hidden.

using Oscar
using StaticArrays
using LinearAlgebra

#==============================================================================#
#                         ABSTRACT TYPE HIERARCHY                               #
#==============================================================================#

"""
    AbstractSignomial{T}

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
abstract type AbstractSignomial{T} end

#==============================================================================#
#                    CONCRETE REPRESENTATIONS                                  #
#==============================================================================#

include("static_signomial.jl")
include("matrix_signomial.jl")

#==============================================================================#
#                         COMMON INTERFACE METHODS                              #
#==============================================================================#

# Iteration
Base.eachindex(f::AbstractSignomial) = Base.OneTo(length(f))

# Exponentiation by Float64
Base.:^(f::AbstractSignomial, r::Float64) = f^rationalize(r)

# Exponentiation by Int
Base.:^(f::AbstractSignomial, n::Int) = f^Base.Rational(n)

#==============================================================================#
#                    HELPER CONSTRUCTORS                                        #
#==============================================================================#

"""
    poly_const(n::Int, c::TropicalSemiringElem, ::Type{T}=Float64) where T

Create a constant polynomial in n variables.
"""
function poly_const(n::Int, c::Oscar.TropicalSemiringElem, ::Type{T} = Float64) where {T}
    return OptimalTropicalPoly([c], [zeros(T, n)], true)
end

"""
    poly_zero(n::Int, R::Oscar.TropicalSemiring, ::Type{T}=Float64) where T

Create the zero polynomial (tropical negative infinity) in n variables.
"""
function poly_zero(n::Int, R, ::Type{T} = Float64) where {T}
    return poly_const(n, zero(R(0)), T)
end

"""
    poly_one(n::Int, R::Oscar.TropicalSemiring, ::Type{T}=Float64) where T

Create the one polynomial (multiplicative identity, tropical zero) in n variables.
"""
function poly_one(n::Int, R, ::Type{T} = Float64) where {T}
    return poly_const(n, one(R(0)), T)
end

"""
    poly_monomial(c::TropicalSemiringElem, exp::Vector{T}) where T

Create a monomial polynomial.
"""
function poly_monomial(c::Oscar.TropicalSemiringElem, exp::Vector{T}) where {T}
    return OptimalTropicalPoly([c], [exp], true)
end

#==============================================================================#
#                    STRING REPRESENTATION                                      #
#==============================================================================#

Base.repr(f::AbstractSignomial) = string(f)

#==============================================================================#
#                    SMART CONSTRUCTOR (AUTO-SELECTS IMPLEMENTATION)           #
#==============================================================================#

# Keyword-argument convenience wrappers (preserves API compatibility)
function OptimalTropicalPoly(
        coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{<:AbstractVector{T}};
        sorted::Bool = false
) where {T}
    return OptimalTropicalPoly(coeffs, [Vector{T}(e) for e in exp_vecs], sorted)
end

function OptimalTropicalPoly(
        coeff_dict::Dict{<:AbstractVector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{<:AbstractVector{T}};
        sorted::Bool = false
) where {T}
    plain_exps = [Vector{T}(e) for e in exp_vecs]
    plain_dict = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}(
        Vector{T}(k) => v for (k, v) in coeff_dict
    )
    return OptimalTropicalPoly(plain_dict, plain_exps, sorted)
end

# Convenience constructor: plain Real coefficients → wrapped in tropical semiring
function OptimalTropicalPoly(
        coeffs::Vector{<:Real},
        exp_vecs::Vector{<:AbstractVector{T}};
        sorted::Bool = false
) where {T}
    R = Oscar.tropical_semiring(max)
    return OptimalTropicalPoly(
        Oscar.TropicalSemiringElem{typeof(max)}[R(c) for c in coeffs],
        [Vector{T}(e) for e in exp_vecs], sorted)
end

function OptimalTropicalPoly(
        coeffs::Vector{<:Real},
        exp_vecs::Vector{<:AbstractVector{T}},
        sorted::Bool
) where {T}
    R = Oscar.tropical_semiring(max)
    return OptimalTropicalPoly(
        Oscar.TropicalSemiringElem{typeof(max)}[R(c) for c in coeffs],
        [Vector{T}(e) for e in exp_vecs], sorted)
end

"""
    OptimalTropicalPoly(coeff, exp, sorted=false)

Construct a tropical Puiseux polynomial from coefficients and exponent vectors.
For dimensions 1-5 this returns a `SignomialStatic`; for larger dimensions it
returns a `SignomialMatrix`.
"""
function OptimalTropicalPoly(
        coeffs::Vector{Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T}
    if isempty(exp_vecs)
        return SignomialMatrix{T}(Matrix{T}(undef, 0, 0), coeffs)
    end

    dim = length(exp_vecs[1])

    # Choose representation based on dimension
    if dim == 1
        return SignomialStatic{T, 1}(coeffs, exp_vecs, sorted)
    elseif dim == 2
        return SignomialStatic{T, 2}(coeffs, exp_vecs, sorted)
    elseif dim == 3
        return SignomialStatic{T, 3}(coeffs, exp_vecs, sorted)
    elseif dim == 4
        return SignomialStatic{T, 4}(coeffs, exp_vecs, sorted)
    elseif dim == 5
        return SignomialStatic{T, 5}(coeffs, exp_vecs, sorted)
    else
        return SignomialMatrix{T}(coeffs, exp_vecs, sorted)
    end
end

# Convenience constructor from Dict
function OptimalTropicalPoly(
        coeff_dict::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}},
        exp_vecs::Vector{Vector{T}},
        sorted::Bool = false
) where {T}
    coeffs = [coeff_dict[e] for e in exp_vecs]
    return OptimalTropicalPoly(coeffs, exp_vecs, sorted)
end

#==============================================================================#
#                    CONVERSION BETWEEN REPRESENTATIONS                         #
#==============================================================================#

# Note: Conversion functions to/from baseline Signomial are defined
# in a separate extension file that loads when TropicalNN is available.
# For standalone use, these representations work independently.

#==============================================================================#
#                    RATIONAL FUNCTIONS                                         #
#==============================================================================#

"""
    RationalSignomial{P<:AbstractSignomial}

Tropical Puiseux rational function, represented as a quotient of two signomials.
"""
struct RationalSignomial{P <: AbstractSignomial}
    num::P
    den::P

    function RationalSignomial(num::P, den::P) where {P <: AbstractSignomial}
        new{P}(num, den)
    end
end

# Constructor from numerator and denominator data
function OptimalTropicalRational(num_coeffs, num_exp, den_coeffs, den_exp, sorted = false)
    num = OptimalTropicalPoly(num_coeffs, num_exp, sorted)
    den = OptimalTropicalPoly(den_coeffs, den_exp, sorted)
    return RationalSignomial(num, den)
end

# Arithmetic
function Base.:+(f::RationalSignomial{P}, g::RationalSignomial{P}) where {P}
    num = f.num * g.den + f.den * g.num
    den = f.den * g.den
    return RationalSignomial(num, den)
end

function Base.:*(f::RationalSignomial{P}, g::RationalSignomial{P}) where {P}
    return RationalSignomial(f.num * g.num, f.den * g.den)
end

function Base.:/(f::RationalSignomial{P}, g::RationalSignomial{P}) where {P}
    return RationalSignomial(f.num * g.den, f.den * g.num)
end

function eval_rational(f::RationalSignomial, a::Vector)
    return eval_poly(f.num, a) / eval_poly(f.den, a)
end

#==============================================================================#
#                              EXPORTS                                          #
#==============================================================================#

export AbstractSignomial
export SignomialStatic, SignomialMatrix
export RationalSignomial
export OptimalTropicalPoly, OptimalTropicalRational
export get_exp, get_coeff, eval_poly, eval_rational
export quicksum, mul_with_quicksum, comp
export poly_const, poly_zero, poly_one, poly_monomial

#==============================================================================#
#                    USER-FACING CONSTRUCTOR ALIAS                              #
#==============================================================================#

"""
    Signomial(coeffs, exp_vecs, sorted=false)
    Signomial(coeff_dict, exp_vecs, sorted=false)

Alias for `OptimalTropicalPoly`.
"""
const Signomial = OptimalTropicalPoly

#==============================================================================#
#                    ADDITIONAL ACCESSOR FUNCTIONS                              #
#==============================================================================#

"""
    monomial_pairs(f::AbstractSignomial)

Return an iterable of `(exp, coeff)` pairs in lexicographic exponent order.
"""
function monomial_pairs(f::AbstractSignomial)
    return zip(exponents(f), coefficients(f))
end

#==============================================================================#
#                    FACTORY CONSTRUCTORS (rat_maps.jl compat)                 #
#==============================================================================#

"""
    Signomial_const(n, c, f::AbstractSignomial{T})

Construct the constant `c` as a signomial in `n` variables.
`f` is used only to infer the exponent numeric type `T`.
"""
function Signomial_const(n, c, f::AbstractSignomial{T}) where {T}
    return OptimalTropicalPoly([c], [zeros(T, n)], true)
end

"""
    Signomial_zero(n, f::AbstractSignomial)

Construct the tropical-zero (additive identity, −∞) signomial in `n` variables.
`f` is used only to infer the exponent type.
"""
function Signomial_zero(n::Int, f::AbstractSignomial)
    return Signomial_const(n, zero(get_coeff(f, 1)), f)
end

"""
    Signomial_one(n, f::AbstractSignomial)

Construct the tropical-one (multiplicative identity, value 0) signomial in `n` variables.
`f` is used only to infer the exponent type.
"""
function Signomial_one(n::Int, f::AbstractSignomial)
    return Signomial_const(n, one(get_coeff(f, 1)), f)
end

"""
    SignomialMonomial(c, exp::Vector{T})

Construct a single-monomial signomial with coefficient `c` and exponent vector `exp`.
"""
function SignomialMonomial(c, exp::Vector{T}) where {T}
    return OptimalTropicalPoly([c], [exp], true)
end

#==============================================================================#
#                    RATIONAL SIGNOMIAL FACTORIES                               #
#==============================================================================#

"""
    signomial_to_rational(f::AbstractSignomial)

Wrap a signomial as a `RationalSignomial` with denominator equal to tropical one.
"""
function signomial_to_rational(f::AbstractSignomial)
    return RationalSignomial(f, Signomial_one(nvars(f), f))
end

"""
    RationalSignomial_identity(n, c)

Return the vector of coordinate projections `[x₁, x₂, ..., xₙ]` as
`RationalSignomial` values. `c` is a tropical semiring element used only to
infer the coefficient type; only `one(c)` is called on it.
"""
function RationalSignomial_identity(n, c)
    output = Vector{RationalSignomial}()
    sizehint!(output, n)
    for i in 1:n
        push!(
            output,
            signomial_to_rational(SignomialMonomial(one(c), [j == i ? 1 : 0 for j in 1:n]))
        )
    end
    return output
end

"""
    RationalSignomial_zero(n, f::RationalSignomial)

Construct the tropical-zero rational signomial in `n` variables.
`f` is used only to infer types.
"""
function RationalSignomial_zero(n::Int, f::RationalSignomial)
    return RationalSignomial(Signomial_zero(n, f.num), Signomial_one(n, f.den))
end

"""
    RationalSignomial_one(n, f)

Construct the tropical-one rational signomial in `n` variables.
`f` must have a `.num` field (i.e. a `RationalSignomial`) used to infer types.
"""
function RationalSignomial_one(n::Int, f)
    return RationalSignomial(Signomial_one(n, f.num), Signomial_one(n, f.num))
end

#==============================================================================#
#                    OPERATOR EXTENSIONS                                        #
#==============================================================================#

# Division: AbstractSignomial / AbstractSignomial  →  RationalSignomial
function Base.:/(f::P, g::P) where {P <: AbstractSignomial}
    return RationalSignomial(f, g)
end

# TropicalSemiringElem × RationalSignomial
function Base.:*(a::Oscar.TropicalSemiringElem, f::RationalSignomial)
    return RationalSignomial(a * f.num, f.den)
end

# val × RationalSignomial where val is numeric
function Base.:*(val::RationalSignomial, a::Oscar.TropicalSemiringElem)
    return RationalSignomial(a * val.num, val.den)
end

#==============================================================================#
#                    nvars FOR RationalSignomial                                #
#==============================================================================#

function Oscar.nvars(f::RationalSignomial)
    return nvars(f.den)
end

#==============================================================================#
#                    EVALUATE ALIASES                                           #
#==============================================================================#

"""
    evaluate(f::AbstractSignomial, a::Vector)

Evaluate the tropical polynomial `f` at the point `a`.
Alias for `eval_poly`.
"""
function evaluate(f::AbstractSignomial, a::Vector)
    return eval_poly(f, a)
end

"""
    evaluate(f::RationalSignomial, a::Vector)

Evaluate the rational tropical function `f` at the point `a`.
"""
function evaluate(f::RationalSignomial, a::Vector)
    return eval_rational(f, a)
end

"""
    evaluate(F::Vector{<:RationalSignomial}, a::Vector)

Evaluate a vector of rational tropical functions at the point `a`.
"""
function evaluate(F::Vector{<:RationalSignomial}, a::Vector)
    return [evaluate(f, a) for f in F]
end

# Callable syntax: f(x) as sugar for evaluate(f, x)
(f::AbstractSignomial)(x::Vector) = evaluate(f, x)
(f::RationalSignomial)(x::Vector) = evaluate(f, x)

#==============================================================================#
#                    DEDUP AND MONOMIAL COUNT                                   #
#==============================================================================#

"""
    dedup_monomials(f::AbstractSignomial)

Remove all monomials with tropical-zero coefficient from `f`.
"""
function dedup_monomials(f::AbstractSignomial)
    tropical_zero = zero(get_coeff(f, 1))
    new_exps = Vector{eltype(exponents(f))}()
    new_coeffs = Oscar.TropicalSemiringElem{typeof(max)}[]
    for (e, c) in monomial_pairs(f)
        if c != tropical_zero
            push!(new_exps, e)
            push!(new_coeffs, c)
        end
    end
    # Convert SVector to plain Vector for the smart constructor
    plain_exps = [Vector(e) for e in new_exps]
    return OptimalTropicalPoly(new_coeffs, plain_exps, true)
end

function dedup_monomials(f::RationalSignomial)
    return RationalSignomial(dedup_monomials(f.num), dedup_monomials(f.den))
end

function dedup_monomials(F::Vector{<:RationalSignomial})
    return [dedup_monomials(f) for f in F]
end

"""
    monomial_count(f::AbstractSignomial)

Return the number of monomials in `f`.
"""
function monomial_count(f::AbstractSignomial)
    return length(f)
end

function monomial_count(f::RationalSignomial)
    return monomial_count(f.num) + monomial_count(f.den)
end

function monomial_count(F::Vector{<:RationalSignomial})
    return sum(monomial_count(f) for f in F)
end

#==============================================================================#
#                    EXPONENTIATION EXTENSIONS (rat_maps.jl compat)            #
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
function Base.:^(a::Oscar.TropicalSemiringElem{typeof(max)}, b::Rational{T}) where {T <:
                                                                                    Integer}
    R = tropical_semiring(max)
    return R(Rational(a) * b)
end

# TropicalSemiringElem ^ Float64
function Base.:^(a::Oscar.TropicalSemiringElem{typeof(max)}, b::Float64)
    R = tropical_semiring(max)
    return R(rationalize(Float64(Rational(a)) * b))
end

# RationalSignomial exponentiation
function Base.:^(f::RationalSignomial, rat::Float64)
    if rat == 0
        return RationalSignomial_one(nvars(f), f)
    else
        return RationalSignomial(f.num^rat, f.den^rat)
    end
end

function Base.:^(f::RationalSignomial, int::Int)
    if int == 0
        return RationalSignomial_one(nvars(f), f)
    else
        return RationalSignomial(f.num^int, f.den^int)
    end
end

function Base.:^(f::RationalSignomial, r::Rational{T}) where {T <: Integer}
    if r == 0
        return RationalSignomial_one(nvars(f), f)
    else
        return RationalSignomial(f.num^r, f.den^r)
    end
end

#==============================================================================#
#                    QUICKSUM EXTENSIONS                                        #
#==============================================================================#

"""
    quicksum(F::Vector{<:AbstractSignomial})

Fallback quicksum for a heterogeneous vector of `AbstractSignomial` values.
Accumulates terms with plain addition.
"""
function quicksum(F::Vector{<:AbstractSignomial})
    isempty(F) && throw(ArgumentError("Cannot quicksum empty vector"))
    result = F[1]
    for i in 2:length(F)
        result = result + F[i]
    end
    return result
end

"""
    mul_with_quicksum(F::Vector{<:AbstractSignomial})

Multiply all polynomials in `F` using quicksum multiplication.
"""
function mul_with_quicksum(F::Vector{<:AbstractSignomial})
    isempty(F) && throw(ArgumentError("Cannot mul_with_quicksum empty vector"))
    result = F[1]
    for i in 2:length(F)
        result = mul_with_quicksum(result, F[i])
    end
    return result
end

"""
    quicksum(F::Vector{<:RationalSignomial})

Tropical sum of a vector of rational signomials: numerator/denominator arithmetic.
"""
function quicksum(F::Vector{<:RationalSignomial})
    isempty(F) && throw(ArgumentError("Cannot quicksum empty vector"))
    denoms = [f.den for f in F]
    den = foldl(mul_with_quicksum, denoms)
    summands = map(Base.eachindex(F)) do i
        others = [denoms[j] for j in Base.eachindex(F) if j != i]
        isempty(others) ? F[i].num :
        mul_with_quicksum(vcat([F[i].num], others))
    end
    return RationalSignomial(quicksum(summands), den)
end

"""
    mul_with_quicksum(F::Vector{<:RationalSignomial})

Multiply all rational signomials in `F` using quicksum multiplication.
"""
function mul_with_quicksum(F::Vector{<:RationalSignomial})
    isempty(F) && throw(ArgumentError("Cannot mul_with_quicksum empty vector"))
    result = F[1]
    for i in 2:length(F)
        result = mul_with_quicksum(result, F[i])
    end
    return result
end

"""
    mul_with_quicksum(f::RationalSignomial, g::RationalSignomial)

Quicksum multiplication of two rational signomials.
"""
function mul_with_quicksum(f::RationalSignomial, g::RationalSignomial)
    return RationalSignomial(
        mul_with_quicksum(f.num, g.num),
        mul_with_quicksum(f.den, g.den)
    )
end

# AbstractSignomial fallback for mul_with_quicksum
function mul_with_quicksum(f::AbstractSignomial, g::AbstractSignomial)
    return f * g  # fall back to regular multiplication
end

#==============================================================================#
#                    ADD / DIV WITH QUICKSUM                                    #
#==============================================================================#

"""
    add_with_quicksum(f::RationalSignomial, g::RationalSignomial)

Add two rational signomials using quicksum operations.
"""
function add_with_quicksum(f::RationalSignomial, g::RationalSignomial)
    num = quicksum([mul_with_quicksum(f.num, g.den), mul_with_quicksum(f.den, g.num)])
    den = mul_with_quicksum(f.den, g.den)
    return RationalSignomial(num, den)
end

"""
    div_with_quicksum(f::RationalSignomial, g::RationalSignomial)

Divide two rational signomials using quicksum operations.
"""
function div_with_quicksum(f::RationalSignomial, g::RationalSignomial)
    num = mul_with_quicksum(f.num, g.den)
    den = mul_with_quicksum(f.den, g.num)
    return RationalSignomial(num, den)
end

#==============================================================================#
#                    COMPOSITION EXTENSIONS                                     #
#==============================================================================#

"""
    comp(f::AbstractSignomial, G::Vector{<:RationalSignomial})

Compose polynomial `f` with the vector of rational signomials `G`:
computes `f(G[1], G[2], ..., G[n])`.
"""
function comp(f::AbstractSignomial, G::Vector{<:RationalSignomial})
    @assert length(G) == nvars(f) "Number of polynomials must match number of variables"
    result = RationalSignomial_zero(nvars(G[1]), G[1])
    for (e, c) in monomial_pairs(f)
        term = RationalSignomial_one(nvars(G[1]), G[1])
        for i in Base.eachindex(G)
            term *= G[i]^e[i]
        end
        result += c * term
    end
    return result
end

"""
    comp(f::RationalSignomial, G::Vector{<:RationalSignomial})

Compose rational signomial `f` with the vector `G`.
"""
function comp(f::RationalSignomial, G::Vector{<:RationalSignomial})
    num = comp(f.num, G)
    den = comp(f.den, G)
    return num / den
end

"""
    comp(F::Vector{<:RationalSignomial}, G::Vector{<:RationalSignomial})

Compose each element of `F` with `G`.
"""
function comp(F::Vector{<:RationalSignomial}, G::Vector{<:RationalSignomial})
    return [comp(f, G) for f in F]
end

"""
    comp_with_quicksum(f::AbstractSignomial, G::Vector{<:RationalSignomial})

Composition using quicksum operations (faster, defers sorting).
"""
function comp_with_quicksum(f::AbstractSignomial, G::Vector{<:RationalSignomial})
    @assert length(G) == nvars(f) "Number of polynomials must match number of variables"
    summands = RationalSignomial[]
    for (e, c) in monomial_pairs(f)
        term = RationalSignomial_one(nvars(G[1]), G[1])
        for i in Base.eachindex(G)
            term = mul_with_quicksum(term, G[i]^e[i])
        end
        push!(summands, c * term)
    end
    return quicksum(summands)
end

"""
    comp_with_quicksum(f::RationalSignomial, G::Vector{<:RationalSignomial})

Compose rational signomial `f` with `G` using quicksum operations.
"""
function comp_with_quicksum(f::RationalSignomial, G::Vector{<:RationalSignomial})
    num = comp_with_quicksum(f.num, G)
    den = comp_with_quicksum(f.den, G)
    return div_with_quicksum(num, den)
end

"""
    comp_with_quicksum(F::Vector{<:RationalSignomial}, G::Vector{<:RationalSignomial})

Compose each element of `F` with `G` using quicksum.
"""
function comp_with_quicksum(F::Vector{<:RationalSignomial}, G::Vector{<:RationalSignomial})
    return [comp_with_quicksum(f, G) for f in F]
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

function Base.show(io::IO, f::AbstractSignomial)
    if length(f) == 0
        print(io, "max()")
        return
    end
    strs = [_monomial_str(get_coeff(f, i), get_exp(f, i)) for i in Base.eachindex(f)]
    print(io, "max(", join(strs, ", "), ")")
end

function Base.show(io::IO, f::RationalSignomial)
    print(io, "(", f.num, ") ⊘ (", f.den, ")")
end

function Base.show(io::IO, F::Vector{<:RationalSignomial})
    for (i, f) in enumerate(F)
        print(io, "f$(_subscript(i)) = ", f, "\n")
    end
end

#==============================================================================#
#                    BASE.ZERO / BASE.ONE (AbstractSignomial type)             #
#==============================================================================#

function Base.zero(::Type{<:AbstractSignomial{T}}, n::Int) where {T}
    R = Oscar.tropical_semiring(max)
    return OptimalTropicalPoly([zero(R(0))], [zeros(T, n)], true)
end

function Base.one(::Type{<:AbstractSignomial{T}}, n::Int) where {T}
    R = Oscar.tropical_semiring(max)
    return OptimalTropicalPoly([one(R(0))], [zeros(T, n)], true)
end
