# Rational functions built from matrix-backed signomials.

#==============================================================================#
#                    RATIONAL FUNCTIONS                                         #
#==============================================================================#

"""
    RationalSignomial{T}

Tropical rational function represented by a quotient of two signomials.

Use `convert(RationalSignomial{S}, f)` to convert the exponent type of `f` to
`S`.
"""
struct RationalSignomial{T}
    num::Signomial{T}
    den::Signomial{T}

    function RationalSignomial(num::Signomial{T}, den::Signomial{T}) where {T}
        new{T}(num, den)
    end
end

Base.convert(::Type{RationalSignomial{T}}, f::RationalSignomial{T}) where {T} = f

function Base.convert(::Type{RationalSignomial{T}}, f::RationalSignomial) where {T}
    return RationalSignomial(convert(Signomial{T}, f.num), convert(Signomial{T}, f.den))
end

"""
    _embed_variables(f::RationalSignomial, block, input_dimension)

Embed the variables of the numerator and denominator of `f` in `block`.
"""
function _embed_variables(
        f::RationalSignomial,
        block::UnitRange{Int},
        input_dimension::Integer
)
    return RationalSignomial(
        _embed_variables(f.num, block, input_dimension),
        _embed_variables(f.den, block, input_dimension)
    )
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

#==============================================================================#
#                    RATIONAL SIGNOMIAL FACTORIES                               #
#==============================================================================#

"""
    signomial_to_rational(f::Signomial)

Return `f` as a `RationalSignomial` with denominator equal to tropical one.
"""
function signomial_to_rational(f::Signomial)
    return RationalSignomial(f, signomial_one(nvars(f), f))
end

"""
    rational_signomial_identity(n, c)

Return the coordinate projections `[x₁, x₂, …, xₙ]`. Use `c` to infer the
coefficient type.
"""
function rational_signomial_identity(n, c)
    output = Vector{RationalSignomial}()
    sizehint!(output, n)
    for i in 1:n
        push!(
            output,
            signomial_to_rational(signomial_monomial(one(c), [j == i ? 1 : 0 for j in 1:n]))
        )
    end
    return output
end

"""
    rational_signomial_zero(n, f::RationalSignomial)

Construct tropical zero in `n` variables. Use `f` to infer the types.
"""
function rational_signomial_zero(n::Int, f::RationalSignomial)
    return RationalSignomial(signomial_zero(n, f.num), signomial_one(n, f.den))
end

"""
    rational_signomial_one(n, f::RationalSignomial)

Construct tropical one in `n` variables. Use `f` to infer the types.
"""
function rational_signomial_one(n::Int, f)
    return RationalSignomial(signomial_one(n, f.num), signomial_one(n, f.num))
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

Evaluate `f` at point `a`.
"""
function evaluate(f::RationalSignomial, a::Vector)
    point = _coerce_evaluation_point(f.num, a)
    return evaluate(f.num, point) / evaluate(f.den, point)
end

"""
    evaluate(F::Vector{<:RationalSignomial}, a::Vector)

Evaluate each function in `F` at point `a`.
"""
function evaluate(F::Vector{<:RationalSignomial}, a::Vector)
    return [evaluate(f, a) for f in F]
end

# Callable syntax: f(x) as sugar for evaluate(f, x)
(f::RationalSignomial)(x::Vector) = evaluate(f, x)

#==============================================================================#
#                    MONOMIAL COUNT                                             #
#==============================================================================#

function monomial_count(f::RationalSignomial)
    return monomial_count(f.num) + monomial_count(f.den)
end

function monomial_count(F::Vector{<:RationalSignomial})
    return sum(monomial_count(f) for f in F)
end

#==============================================================================#
#                    EXPONENTIATION EXTENSIONS                                  #
#==============================================================================#

Base.inv(f::RationalSignomial) = RationalSignomial(f.den, f.num)

Base.:^(f::RationalSignomial, r::AbstractFloat) = f^rationalize(r)

function Base.:^(f::RationalSignomial, n::Integer)
    if iszero(n)
        return rational_signomial_one(nvars(f), f)
    elseif n < 0
        magnitude = -n
        return RationalSignomial(f.den^magnitude, f.num^magnitude)
    else
        return RationalSignomial(f.num^n, f.den^n)
    end
end

function Base.:^(f::RationalSignomial, r::Rational{T}) where {T <: Integer}
    if r == 0
        return rational_signomial_one(nvars(f), f)
    elseif r < 0
        magnitude = -r
        return RationalSignomial(f.den^magnitude, f.num^magnitude)
    else
        return RationalSignomial(f.num^r, f.den^r)
    end
end

#==============================================================================#
#                    QUICKSUM EXTENSIONS                                        #
#==============================================================================#

"""
    quicksum(F::Vector{<:RationalSignomial})

Return the tropical sum of `F`.
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
#                    COMPOSITION EXTENSIONS                                     #
#==============================================================================#

function _composition_term(e, c, G, one_value)
    term = one_value
    for i in Base.eachindex(G)
        term *= G[i]^e[i]
    end
    return c * term
end

"""
    comp(f::Signomial, G::Vector{<:RationalSignomial}; quicksum=false)

Substitute `G[i]` for variable `i` in `f`.
Set `quicksum=true` to batch the intermediate tropical sums.
"""
function comp(
        f::Signomial,
        G::Vector{<:RationalSignomial};
        quicksum::Bool = false
)
    @assert length(G) == nvars(f) "Number of polynomials must match number of variables"
    zero_value = rational_signomial_zero(nvars(G[1]), G[1])
    one_value = rational_signomial_one(nvars(G[1]), G[1])
    terms = (
        _composition_term(e, c, G, one_value)
    for (e, c) in monomial_pairs(f)
    )
    if quicksum
        collected_terms = collect(terms)
        isempty(collected_terms) && return zero_value
        return TropicalNN.quicksum(collected_terms)
    end
    return foldl(+, terms; init = zero_value)
end

"""
    comp(f::RationalSignomial, G::Vector{<:RationalSignomial}; quicksum=false)

Substitute `G[i]` for variable `i` in `f`.
Set `quicksum=true` to batch the intermediate tropical sums.
"""
function comp(
        f::RationalSignomial,
        G::Vector{<:RationalSignomial};
        quicksum::Bool = false
)
    num = comp(f.num, G; quicksum = quicksum)
    den = comp(f.den, G; quicksum = quicksum)
    return num / den
end

"""
    comp(F::Vector{<:RationalSignomial}, G::Vector{<:RationalSignomial}; quicksum=false)

Substitute `G` into each element of `F`.
Set `quicksum=true` to batch the intermediate tropical sums.
"""
function comp(
        F::Vector{<:RationalSignomial},
        G::Vector{<:RationalSignomial};
        quicksum::Bool = false
)
    return [comp(f, G; quicksum = quicksum) for f in F]
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
