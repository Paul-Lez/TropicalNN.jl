# Rational functions built from matrix-backed signomials.

#==============================================================================#
#                    RATIONAL FUNCTIONS                                         #
#==============================================================================#

"""
    RationalSignomial{T}

Tropical Puiseux rational function, represented as a quotient of two signomials.
"""
struct RationalSignomial{T}
    num::Signomial{T}
    den::Signomial{T}

    function RationalSignomial(num::Signomial{T}, den::Signomial{T}) where {T}
        new{T}(num, den)
    end
end

# Constructor from numerator and denominator data
function OptimalTropicalRational(num_coeffs, num_exp, den_coeffs, den_exp, sorted = false)
    num = Signomial(num_coeffs, num_exp; sorted = sorted)
    den = Signomial(den_coeffs, den_exp; sorted = sorted)
    return RationalSignomial(num, den)
end

# Arithmetic
function Base.:+(f::RationalSignomial{T}, g::RationalSignomial{T}) where {T}
    num = f.num * g.den + f.den * g.num
    den = f.den * g.den
    return RationalSignomial(num, den)
end

function Base.:*(f::RationalSignomial{T}, g::RationalSignomial{T}) where {T}
    return RationalSignomial(f.num * g.num, f.den * g.den)
end

function Base.:/(f::RationalSignomial{T}, g::RationalSignomial{T}) where {T}
    return RationalSignomial(f.num * g.den, f.den * g.num)
end

function eval_rational(f::RationalSignomial, a::Vector)
    return eval_poly(f.num, a) / eval_poly(f.den, a)
end

#==============================================================================#
#                    RATIONAL SIGNOMIAL FACTORIES                               #
#==============================================================================#

"""
    signomial_to_rational(f::Signomial)

Wrap a signomial as a `RationalSignomial` with denominator equal to tropical one.
"""
function signomial_to_rational(f::Signomial)
    return RationalSignomial(f, Signomial_one(nvars(f), f))
end

"""
    RationalSignomial_identity(n, c)

Return the vector of coordinate projections `[x1, x2, ..., xn]` as
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

# Division: Signomial / Signomial -> RationalSignomial
function Base.:/(f::Signomial{T}, g::Signomial{T}) where {T}
    return RationalSignomial(f, g)
end

# TropicalSemiringElem x RationalSignomial
function Base.:*(a::Oscar.TropicalSemiringElem, f::RationalSignomial)
    return RationalSignomial(a * f.num, f.den)
end

# RationalSignomial x TropicalSemiringElem
function Base.:*(val::RationalSignomial, a::Oscar.TropicalSemiringElem)
    return RationalSignomial(a * val.num, val.den)
end

function Oscar.nvars(f::RationalSignomial)
    return nvars(f.den)
end

#==============================================================================#
#                    EVALUATE ALIASES                                           #
#==============================================================================#

"""
    evaluate(f::RationalSignomial, a::Vector)

Evaluate the rational tropical function `f` at the point `a`.
"""
function evaluate(f::RationalSignomial, a::Vector)
    return eval_rational(f, _coerce_evaluation_point(f.num, a))
end

"""
    evaluate(F::Vector{<:RationalSignomial}, a::Vector)

Evaluate a vector of rational tropical functions at the point `a`.
"""
function evaluate(F::Vector{<:RationalSignomial}, a::Vector)
    return [evaluate(f, a) for f in F]
end

# Callable syntax: f(x) as sugar for evaluate(f, x)
(f::RationalSignomial)(x::Vector) = evaluate(f, x)

#==============================================================================#
#                    DEDUP AND MONOMIAL COUNT                                   #
#==============================================================================#

function dedup_monomials(f::RationalSignomial)
    return RationalSignomial(dedup_monomials(f.num), dedup_monomials(f.den))
end

function dedup_monomials(F::Vector{<:RationalSignomial})
    return [dedup_monomials(f) for f in F]
end

function monomial_count(f::RationalSignomial)
    return monomial_count(f.num) + monomial_count(f.den)
end

function monomial_count(F::Vector{<:RationalSignomial})
    return sum(monomial_count(f) for f in F)
end

#==============================================================================#
#                    EXPONENTIATION EXTENSIONS                                  #
#==============================================================================#

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
    quicksum(F::Vector{<:RationalSignomial})

Tropical sum of a vector of rational signomials: numerator/denominator arithmetic.
"""
function quicksum(F::Vector{<:RationalSignomial})
    isempty(F) && throw(ArgumentError("Cannot quicksum empty vector"))
    denoms = [f.den for f in F]
    den = foldl(*, denoms)
    summands = map(Base.eachindex(F)) do i
        others = [denoms[j] for j in Base.eachindex(F) if j != i]
        isempty(others) ? F[i].num :
        foldl(*, vcat([F[i].num], others))
    end
    return RationalSignomial(quicksum(summands), den)
end

#==============================================================================#
#                    ADD / DIV WITH QUICKSUM                                    #
#==============================================================================#

"""
    add_with_quicksum(f::RationalSignomial, g::RationalSignomial)

Add two rational signomials using quicksum addition for the numerator.
"""
function add_with_quicksum(f::RationalSignomial, g::RationalSignomial)
    num = quicksum([f.num * g.den, f.den * g.num])
    den = f.den * g.den
    return RationalSignomial(num, den)
end

"""
    div_with_quicksum(f::RationalSignomial, g::RationalSignomial)

Divide two rational signomials.
"""
function div_with_quicksum(f::RationalSignomial, g::RationalSignomial)
    num = f.num * g.den
    den = f.den * g.num
    return RationalSignomial(num, den)
end

#==============================================================================#
#                    COMPOSITION EXTENSIONS                                     #
#==============================================================================#

"""
    comp(f::Signomial, G::Vector{<:RationalSignomial})

Compose polynomial `f` with the vector of rational signomials `G`:
computes `f(G[1], G[2], ..., G[n])`.
"""
function comp(f::Signomial, G::Vector{<:RationalSignomial})
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
    comp_with_quicksum(f::Signomial, G::Vector{<:RationalSignomial})

Composition using exact quicksum operations, which batch intermediate sums where available.
"""
function comp_with_quicksum(f::Signomial, G::Vector{<:RationalSignomial})
    @assert length(G) == nvars(f) "Number of polynomials must match number of variables"
    summands = RationalSignomial[]
    for (e, c) in monomial_pairs(f)
        term = RationalSignomial_one(nvars(G[1]), G[1])
        for i in Base.eachindex(G)
            term *= G[i]^e[i]
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
    return num / den
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

function Base.show(io::IO, f::RationalSignomial)
    print(io, "(", f.num, ") ⊘ (", f.den, ")")
end

function Base.show(io::IO, F::Vector{<:RationalSignomial})
    for (i, f) in enumerate(F)
        print(io, "f$(_subscript(i)) = ", f, "\n")
    end
end
