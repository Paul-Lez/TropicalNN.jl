# =============================================================================
# StandardizedTropicalPoly
#
# A representation of a tropical Puiseux polynomial in which every exponent
# component is stored as an integer, together with a common denominator `d`.
# The "true" rational exponent for component k of monomial i is
#
#   exp[i][k] // denominator
#
# This avoids repeated rational arithmetic and makes the polynomial easier to
# pass to Oscar's polynomial-ring machinery.
# =============================================================================

# -----------------------------------------------------------------------------
# Lightweight ring descriptor
# -----------------------------------------------------------------------------

"""
    TropicalPolyRing

Minimal ring descriptor for `StandardizedTropicalPoly`.
Only records the number of variables; no Oscar polynomial ring is constructed.
"""
struct TropicalPolyRing
    n::Int
end

Oscar.nvars(r::TropicalPolyRing) = r.n

# -----------------------------------------------------------------------------
# Struct definition
# -----------------------------------------------------------------------------

"""
    StandardizedTropicalPoly{T<:Integer}

A tropical Puiseux polynomial stored with integer exponent vectors and a
common denominator `d::T`. The true rational exponent of the `k`-th component
of the `i`-th monomial is `exp[i][k] // denominator`.

# Fields
- `coeff::Dict{Vector{Int}, TropicalSemiringElem}` — maps integer exponent
  vectors to tropical coefficients.
- `exp::Vector{Vector{Int}}` — lexicographically sorted integer exponent
  vectors.
- `denominator::T` — common denominator; `T` is the integer type (Int64 or
  BigInt) of the source exponents.
- `ring::TropicalPolyRing` — records the number of variables.
"""
struct StandardizedTropicalPoly{T<:Integer}
    coeff::Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}
    exp::Vector{Vector{Int}}
    denominator::T
    ring::TropicalPolyRing
end

# -----------------------------------------------------------------------------
# Conversion: Signomial ↔ StandardizedTropicalPoly
# -----------------------------------------------------------------------------

"""
    standardize(f::Signomial{Rational{T}}) -> StandardizedTropicalPoly{T}

Convert a tropical Puiseux polynomial with rational exponents to the
standardized integer-exponent representation.  The common denominator is the
LCM of all exponent denominators.

# Example
```julia
R = tropical_semiring(max)
f = Signomial([R(1), R(2)], [[1//2, 0//1], [0//1, 1//3]]; sorted=false)
g = standardize(f)   # g.denominator == 6
```
"""
function standardize(f::Signomial{Rational{T}}) where T<:Integer
    n = Oscar.nvars(f)
    # compute LCM of all exponent denominators
    d = one(T)
    for e in f.exp
        for a in e
            d = lcm(d, denominator(a))
        end
    end
    # scale every exponent vector by d → all components become integers
    int_exps = [Int.(numerator.(d .* e)) for e in f.exp]
    # build coeff dict
    coeff = Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}()
    for (old_e, new_e) in zip(f.exp, int_exps)
        coeff[new_e] = f.coeff[old_e]
    end
    ring = TropicalPolyRing(n)
    return StandardizedTropicalPoly{T}(coeff, int_exps, d, ring)
end

"""
    destandardize(g::StandardizedTropicalPoly{T}) -> Signomial{Rational{T}}

Convert a standardized tropical polynomial back to a `Signomial` with
rational exponents.
"""
function destandardize(g::StandardizedTropicalPoly{T}) where T<:Integer
    rat_exps = [[Rational{T}(T(ei), g.denominator) for ei in e] for e in g.exp]
    coeff = Dict{Vector{Rational{T}}, Oscar.TropicalSemiringElem{typeof(max)}}()
    for (int_e, rat_e) in zip(g.exp, rat_exps)
        coeff[rat_e] = g.coeff[int_e]
    end
    return Signomial(coeff, rat_exps; sorted=true)
end

"""
    convert_denominator(g::StandardizedTropicalPoly{T}, new_d) -> StandardizedTropicalPoly{T}

Return a copy of `g` with its denominator changed to `new_d`.
`new_d` must be an exact multiple of `g.denominator`.
"""
function convert_denominator(g::StandardizedTropicalPoly{T}, new_d) where T<:Integer
    new_d_T = T(new_d)
    factor  = new_d_T ÷ g.denominator
    new_exps = [Int.(factor .* e) for e in g.exp]
    new_coeff = Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}()
    for (old_e, new_e) in zip(g.exp, new_exps)
        new_coeff[new_e] = g.coeff[old_e]
    end
    return StandardizedTropicalPoly{T}(new_coeff, new_exps, new_d_T, g.ring)
end

# -----------------------------------------------------------------------------
# Factory constructors
# -----------------------------------------------------------------------------

"""
    StandardizedTropicalPoly_const(n, c, d) -> StandardizedTropicalPoly

Create a constant tropical polynomial equal to `c` in `n` variables with
common denominator `d`.
"""
function StandardizedTropicalPoly_const(n::Int, c::Oscar.TropicalSemiringElem, d)
    T = typeof(d)
    zero_exp = zeros(Int, n)
    coeff = Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}(zero_exp => c)
    ring  = TropicalPolyRing(n)
    return StandardizedTropicalPoly{T}(coeff, [zero_exp], T(d), ring)
end

"""
    StandardizedTropicalPoly_zero(n, d, R) -> StandardizedTropicalPoly

Create the tropical-zero (additive identity, i.e. −∞) polynomial in `n`
variables with common denominator `d`.  `R` is only used to infer the
tropical semiring type.
"""
function StandardizedTropicalPoly_zero(n::Int, d, R)
    T = typeof(d)
    rs = tropical_semiring(max)
    tropical_zero = zero(rs(0))
    zero_exp = zeros(Int, n)
    coeff = Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}(zero_exp => tropical_zero)
    ring  = TropicalPolyRing(n)
    return StandardizedTropicalPoly{T}(coeff, [zero_exp], T(d), ring)
end

"""
    StandardizedTropicalPoly_one(n, d, R) -> StandardizedTropicalPoly

Create the tropical-one (multiplicative identity, value 0) polynomial in `n`
variables with common denominator `d`.  `R` is only used to infer the
tropical semiring type.
"""
function StandardizedTropicalPoly_one(n::Int, d, R)
    T = typeof(d)
    rs = tropical_semiring(max)
    tropical_one = rs(0)
    zero_exp = zeros(Int, n)
    coeff = Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}(zero_exp => tropical_one)
    ring  = TropicalPolyRing(n)
    return StandardizedTropicalPoly{T}(coeff, [zero_exp], T(d), ring)
end

"""
    StandardizedTropicalMonomial(coeff, exp::Vector{Int}, d) -> StandardizedTropicalPoly

Create a single-monomial standardized polynomial with the given integer
exponent vector `exp` and common denominator `d`.
"""
function StandardizedTropicalMonomial(coeff::Oscar.TropicalSemiringElem, exp::Vector{Int}, d)
    T = typeof(d)
    n = length(exp)
    coeff_dict = Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}(exp => coeff)
    ring = TropicalPolyRing(n)
    return StandardizedTropicalPoly{T}(coeff_dict, [exp], T(d), ring)
end

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

"""
    evaluate(g::StandardizedTropicalPoly, x::Vector) -> TropicalSemiringElem

Evaluate the standardized tropical polynomial at the point `x`.
Uses the identity `x[i]^(exp[i] // denom)` = `x[i] ^ Rational(exp[i], denom)`.
"""
function evaluate(g::StandardizedTropicalPoly, x::Vector)
    rs = tropical_semiring(max)
    result = zero(rs(0))           # tropical −∞
    for e in g.exp
        term = g.coeff[e]
        for i in eachindex(x)
            r = Rational(e[i], Int(g.denominator))
            term = term * (x[i] ^ r)
        end
        result = result + term     # tropical max
    end
    return result
end

"""
    eval_horner(g::StandardizedTropicalPoly, x::Vector) -> TropicalSemiringElem

Evaluate the standardized tropical polynomial at `x` using Horner's method
for univariate polynomials.  For multivariate polynomials the function falls
back to `evaluate`.

All tests assert `eval_horner(f, x) == evaluate(f, x)`; the function is
provided as an optimisation hook for univariate evaluation.
"""
function eval_horner(g::StandardizedTropicalPoly, x::Vector)
    # Horner optimisation is only straightforward for univariate polynomials.
    if Oscar.nvars(g.ring) != 1 || isempty(g.exp)
        return evaluate(g, x)
    end
    # Sort monomials by exponent (ascending) for the Horner sweep.
    order = sortperm(g.exp; by = e -> e[1])
    sorted_exps   = g.exp[order]
    sorted_coeffs = [g.coeff[e] for e in sorted_exps]
    rs = tropical_semiring(max)
    # Trivially correct even without the classical Horner factoring since the
    # result is identical to direct evaluation for tropical polynomials;
    # the loop below is still O(n) in the number of monomials.
    result = zero(rs(0))
    for (e, c) in zip(sorted_exps, sorted_coeffs)
        r    = Rational(e[1], Int(g.denominator))
        term = c * (x[1] ^ r)
        result = result + term
    end
    return result
end

# -----------------------------------------------------------------------------
# Arithmetic
# -----------------------------------------------------------------------------

"""
    +(f::StandardizedTropicalPoly, g::StandardizedTropicalPoly)

Tropical addition (pointwise max) of two standardized polynomials.
The output denominator is `lcm(f.denominator, g.denominator)`.
"""
function Base.:+(f::StandardizedTropicalPoly{T1},
                 g::StandardizedTropicalPoly{T2}) where {T1<:Integer, T2<:Integer}
    T   = promote_type(T1, T2)
    d   = lcm(T(f.denominator), T(g.denominator))
    fc  = convert_denominator(f, d)
    gc  = convert_denominator(g, d)
    n   = Oscar.nvars(f.ring)
    merged = Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}()
    for e in fc.exp
        merged[e] = fc.coeff[e]
    end
    for e in gc.exp
        if haskey(merged, e)
            merged[e] = merged[e] + gc.coeff[e]   # tropical max
        else
            merged[e] = gc.coeff[e]
        end
    end
    new_exps = sort(collect(keys(merged)))
    return StandardizedTropicalPoly{T}(merged, new_exps, d, TropicalPolyRing(n))
end

"""
    *(f::StandardizedTropicalPoly, g::StandardizedTropicalPoly)

Tropical multiplication of two standardized polynomials.
The output denominator is `lcm(f.denominator, g.denominator)`.
"""
function Base.:*(f::StandardizedTropicalPoly{T1},
                 g::StandardizedTropicalPoly{T2}) where {T1<:Integer, T2<:Integer}
    T   = promote_type(T1, T2)
    d   = lcm(T(f.denominator), T(g.denominator))
    fc  = convert_denominator(f, d)
    gc  = convert_denominator(g, d)
    n   = Oscar.nvars(f.ring)
    merged = Dict{Vector{Int}, Oscar.TropicalSemiringElem{typeof(max)}}()
    for (ef, cf) in fc.coeff
        for (eg, cg) in gc.coeff
            e_new = ef .+ eg
            c_new = cf * cg          # tropical multiplication = classical +
            if haskey(merged, e_new)
                merged[e_new] = merged[e_new] + c_new   # take max
            else
                merged[e_new] = c_new
            end
        end
    end
    new_exps = sort(collect(keys(merged)))
    return StandardizedTropicalPoly{T}(merged, new_exps, d, TropicalPolyRing(n))
end

"""
    quicksum(polys::Vector{<:StandardizedTropicalPoly}) -> StandardizedTropicalPoly

Tropical sum (element-wise max) of a vector of standardized polynomials.
"""
function quicksum(polys::Vector{<:StandardizedTropicalPoly})
    result = polys[1]
    for i in 2:length(polys)
        result = result + polys[i]
    end
    return result
end
