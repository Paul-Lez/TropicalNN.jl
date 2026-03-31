####################### BASIC CONSTRUCTIONS ##################################

"""
Represents a tropical Puiseux polynomial, i.e. a tropical polynomial in several variables, whose
exponents might be rational numbers (i.e. we use this structure when T is a subtype of the rational numbers).
The coefficients are elements of the tropical semiring.

# Example

julia> f = Signomial(Dict([1, 2] => 1, [2, 1] => 2), [[1, 2], [2, 1]])
  Signomial{Int64}(Dict([2, 1] => 2, [1, 2] => 1), [[1, 2], [2, 1]])

"""
struct Signomial{T}
    coeff::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}
    exp::Vector{Vector{T}}

    # Inner constructor to handle type conversion
    function Signomial{T}(coeff::Dict, exp::Vector{Vector{T}}) where T
        R = Oscar.tropical_semiring(max)
        typed_coeff = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}()
        for (k, v) in coeff
            typed_coeff[k] = v isa Oscar.TropicalSemiringElem ? v : R(v)
        end
        new{T}(typed_coeff, exp)
    end

    # Direct constructor for already-typed dicts
    function Signomial{T}(coeff::Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}, exp::Vector{Vector{T}}) where T
        new{T}(coeff, exp)
    end
end

"""
Represents a quotient of signomials.
"""
struct RationalSignomial{T}
    num::Signomial{T}
    den::Signomial{T}
end

@doc raw"""
    Signomial(coeff::Dict, exp::Vector{Vector{T}}; sorted::Bool=false)

Constructs a signomial from a dictionary of coefficients and a vector of exponents.
Pass `sorted=true` only if the exponents are already in lexicographic order.
"""
# 3-arg positional helper: sorts if needed, then dispatches to the inner {T} constructor.
# This is the final sorting gateway — it does NOT call back through any outer constructor.
function Signomial(coeff::Dict, exp::Vector{Vector{T}}, sorted::Bool) where T
    if !sorted
        exp = sort(exp)
    end
    return Signomial{T}(coeff, exp)   # call inner parameterised constructor directly
end

# Keyword convenience wrapper — calling Signomial(coeff, exp; sorted=true/false)
# dispatches here, then on to the 3-arg positional above.
Signomial(coeff::Dict, exp::Vector{Vector{T}}; sorted::Bool=false) where T =
    Signomial(coeff, exp, sorted)

@doc raw"""
    Signomial(coeff::Vector, exp::Vector; sorted::Bool=false)

Constructs a signomial from a vector of coefficients and a vector of exponents.
If `sorted=false` (the default), the exponents are sorted lexicographically and
coefficients reordered to match. Pass `sorted=true` only if the exponents are
already in lexicographic order.

```jldoctest
julia> f = Signomial([1, 2], [[1, 2], [2, 1]])
  Signomial{Int64}(Dict([2, 1] => 2, [1, 2] => 1), [[1, 2], [2, 1]])
```
"""
function Signomial(coeff::Vector, exp::Vector; sorted::Bool=false)
    if !sorted
        I = sortperm(exp)
        exp = exp[I]
        coeff = coeff[I]
    end
    # Pass sorted=true since we just sorted (or confirmed already sorted)
    return Signomial(Dict(zip(exp, coeff)), exp, true)
end

@doc raw"""
    Signomial(coeff::Vector{<:Real}, exp::Vector; sorted::Bool=false)

Convenience constructor that accepts plain numbers (e.g. `Int`, `Float64`) as
coefficients, automatically wrapping them in the max-plus tropical semiring.

# Example
```julia
# Equivalent to the Oscar-based construction but without ceremony:
f = Signomial([0, 1, -1], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]])
```
"""
function Signomial(coeff::Vector{<:Real}, exp::Vector; sorted::Bool=false)
    R = Oscar.tropical_semiring(max)
    return Signomial(R.(coeff), exp; sorted=sorted)
end

@doc raw"""
    Signomial_const(n, c, f::Signomial{T})

Outputs the constant c viewed as a signomial in n variables, and exponents in the
same type as f.
"""
function Signomial_const(n, c, f::Signomial{T}) where T
    exp = [Base.zeros(T, n)]
    coeff = Dict(Base.zeros(T, n) => c)
    return Signomial(coeff, exp)
end

@doc raw"""
    Signomial_zero(n, f::Signomial{T})
Outputs the tropical zero viewed as a signomial in n variables, and exponents in the
same type as f.
"""
function Signomial_zero(n::Int64, f::Signomial{T}) where T
    return Signomial_const(n, zero(f.coeff[f.exp[1]]), f)
end

@doc raw"""
    Signomial_one(n, f::Signomial{T})
Outputs the tropical one viewed as a signomial in n variables, and exponents in the
same type as f.
"""
function Signomial_one(n::Int64, f::Signomial{T}) where T
    return Signomial_const(n, one(f.coeff[f.exp[1]]), f)
end

@doc raw"""
    SignomialMonomial(c, exp::Vector{T})

Constructs a signomial from a scalar c and a vector of exponents. This is a monomial whose
coefficient is c and exponents are given by exp.
"""
function SignomialMonomial(c, exp::Vector{T}) where T
    return Signomial([c], [exp]; sorted=true)
end

@doc raw"""
    signomial_to_rational(f)

Constructs a rational signomial from a signomial f,
by setting the denominator to be the tropical one.
"""
function signomial_to_rational(f)
    return RationalSignomial(f, Signomial_one(nvars(f), f))
end

@doc raw"""
The identity function viewed as a rational signomial in n variables.
"""
function RationalSignomial_identity(n, c)
    output = Vector{RationalSignomial}()
    sizehint!(output, n)
    for i in 1:n
        # add the i-th coordinate viewed as a tropical rational function
        push!(output, signomial_to_rational(
            SignomialMonomial(one(c), [j == i ? 1 : 0 for j in 1:n])))
    end
    return output
end

@doc raw"""
Returns an iterator for the exponents of a signomial.
"""
Base.eachindex(f::Signomial) = Base.eachindex(f.exp)

@doc raw"""
Returns the number of variables of a signomial.
"""
function Oscar.nvars(f::Signomial)
    if !is_empty(f.coeff)
        return length(f.exp[1])
    else
        throw(ArgumentError("nvars is not defined for an empty Signomial"))
    end
end

@doc raw"""
Returns the number of variables of a rational signomial.
"""
function Oscar.nvars(f::RationalSignomial)
    return Oscar.nvars(f.den)
end

@doc raw"""
Outputs zero, viewed as a rational signomial in n variables, and with exponents in the same type as f.
"""
function RationalSignomial_zero(n::Int64, f::RationalSignomial{T}) where T
    return RationalSignomial(Signomial_zero(n, f.num), Signomial_one(n, f.den))
end

@doc raw"""
Outputs one, viewed as a rational signomial in n variables, and with exponents in the same type as f.
"""
function RationalSignomial_one(n::Int64, f)
    return RationalSignomial(Signomial_one(n, f.num), Signomial_one(n, f.num))
end

##################################################################################

####################### STRING REPRESENTATIONS ###################################

function Base.string(f::Signomial{T}) where T
    str = ""
    for i in eachindex(f)
        # in dimension 1 we omit subscripts on the variables
        if nvars(f)==1
            if i == 1
                str *= repr(f.coeff[f.exp[i]]) * "*T^" * repr(f.exp[i][1])
            else
                str *= " + " * repr(f.coeff[f.exp[i]]) * "*T^" * repr(f.exp[i][1])
            end
        else
            if i == 1
                str *= repr(f.coeff[f.exp[i]])
            else
                str *= " + " * repr(f.coeff[f.exp[i]])
            end
            exp = f.exp[i]
            for j in Base.eachindex(exp)
                str *= " * T_" * repr(j) * " ^ " * repr(exp[j])
            end
        end
    end
    return str
end

function Base.repr(f::Signomial)
    return string(f)
end

function Base.string(f::RationalSignomial)
    return string(f.num) * " / " * string(f.den)
end

######################################################################

########################## EVALUATION ################################

#### This section defines API to evaluate Signomials ####

@doc raw"""
    evaluate(f::Signomial, a::Vector)
Evaluates the signomial f at the point a.
"""
function evaluate(f::Signomial{T}, a::Vector) where T
    #R = tropical_semiring(max)
    ev = zero(a[1])
    for (exp, coeff) in f.coeff
        term = one(a[1])
        for i in Base.eachindex(a)
            term *= a[i]^exp[i]
        end
        ev += coeff * term
    end
    return ev
end

@doc raw"""
    evaluate(f::RationalSignomial{T}, a::Vector)
Evaluates the rational signomial f at the point a.
"""
function evaluate(f::RationalSignomial{T}, a::Vector) where T
    n::TropicalSemiringElem{typeof(max)} = evaluate(f.num, a)
    m::TropicalSemiringElem{typeof(max)} = evaluate(f.den, a)
    return n / m
end

@doc raw"""
    evaluate(F::Vector{RationalSignomial{T}}, a::Vector)
Evaluates the vector of rational signomials F at the point a.
"""
function evaluate(F::Vector{RationalSignomial{T}}, a::Vector) where T
    return [evaluate(f, a) for f in F]
end

######################################################################

################ ARITHMETIC OPERATIONS ###############################

#### This section implements standard arithmetic operations for
#### signomials and rational signomials

function Base.:/(f::Signomial{T}, g::Signomial{T}) where T
    return RationalSignomial(f, g)
end

"""
    quicksum(F::Vector{Signomial{T}})

Faster alternative to iterated `+` when summing many polynomials. Instead of O(n) pairwise
sorted merges, collects all terms and sorts once at the end.

!!! warning
    The intermediate `h_exp` accumulator is unsorted. The final `Signomial` constructor call
    passes `sorted=false` so the result is always correctly sorted. Do **not** change this to
    `sorted=true` — that would skip sorting and silently corrupt any downstream `+` operation.
"""
function quicksum(F::Vector{Signomial{T}}) where T
    # Estimate total number of terms
    total_terms = sum(length(f.exp) for f in F)

    # Pre-allocate with proper types
    h_coeff = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(h_coeff, total_terms)
    h_exp = Vector{Vector{T}}()
    sizehint!(h_exp, total_terms)

    # Collect all exponents
    @inbounds for f in F
        for exp in f.exp
            push!(h_exp, exp)
        end
    end

    # Sum coefficients for each unique exponent
    @inbounds for exp in h_exp
        if !haskey(h_coeff, exp)
            # First time seeing this exponent, sum from all polynomials
            coeff_sum = zero(F[1].coeff[F[1].exp[1]])
            for f in F
                if haskey(f.coeff, exp)
                    coeff_sum += f.coeff[exp]
                end
            end
            h_coeff[exp] = coeff_sum
        end
    end

    return Signomial(h_coeff, h_exp; sorted=false)
end

"""
Takes two Signomials whose exponents are lexicographically ordered and outputs the sum with
lexicographically ordered exponents
"""
function Base.:+(f::Signomial{T}, g::Signomial{T}) where T
    lf = length(f.exp)
    lg = length(g.exp)

    # Pre-allocate result storage
    h_coeff = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(h_coeff, lf + lg)
    h_exp = Vector{Vector{T}}()
    sizehint!(h_exp, lf + lg)

    # Cache zero for comparison
    trop_zero = zero(first(values(f.coeff)))

    # Merge the two sorted exponent lists
    j = 1
    @inbounds for i in 1:lg
        c = g.exp[i]
        g_coeff_c = g.coeff[c]
        added = false

        # Process f terms that come before c
        while j <= lf
            d = f.exp[j]
            if d > c
                # c < d, add g term if non-zero
                if g_coeff_c != trop_zero
                    h_coeff[c] = g_coeff_c
                    push!(h_exp, c)
                    added = true
                end
                break
            elseif c == d
                # Equal exponents, add both
                f_coeff_d = f.coeff[d]
                sum_coeff = f_coeff_d + g_coeff_c
                if sum_coeff != trop_zero
                    h_coeff[c] = sum_coeff
                    push!(h_exp, c)
                    added = true
                end
                j += 1
                break
            else  # d < c
                # Add f term if non-zero
                f_coeff_d = f.coeff[d]
                if f_coeff_d != trop_zero
                    h_coeff[d] = f_coeff_d
                    push!(h_exp, d)
                end
                j += 1
            end
        end

        # If we exhausted f terms, add remaining g term
        if !added && j > lf && g_coeff_c != trop_zero
            h_coeff[c] = g_coeff_c
            push!(h_exp, c)
        end
    end

    # Add remaining f terms
    @inbounds while j <= lf
        d = f.exp[j]
        f_coeff_d = f.coeff[d]
        if f_coeff_d != trop_zero
            h_coeff[d] = f_coeff_d
            push!(h_exp, d)
        end
        j += 1
    end

    return Signomial(h_coeff, h_exp; sorted=true)
end

"""
Takes two Signomials whose exponents are lexicographically ordered and outputs the product with
lexicographically ordered exponents
"""
function Base.:*(f::Signomial{T}, g::Signomial{T}) where T
    # Pre-allocate result storage
    n_f = length(f.exp)
    n_g = length(g.exp)
    max_terms = n_f * n_g

    result_coeff = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(result_coeff, max_terms)

    # Compute all products directly
    @inbounds for i in 1:n_g
        g_exp_i = g.exp[i]
        g_coeff_i = g.coeff[g_exp_i]
        for j in 1:n_f
            f_exp_j = f.exp[j]
            f_coeff_j = f.coeff[f_exp_j]

            # Compute new exponent
            new_exp = g_exp_i .+ f_exp_j

            # Add or update coefficient (tropical addition is max)
            if haskey(result_coeff, new_exp)
                result_coeff[new_exp] += g_coeff_i * f_coeff_j
            else
                result_coeff[new_exp] = g_coeff_i * f_coeff_j
            end
        end
    end

    # Sort exponents lexicographically
    result_exp = sort(collect(keys(result_coeff)))

    return Signomial(result_coeff, result_exp; sorted=true)
end

# Multiplication of signomials, collecting all pairwise products unsorted and sorting once at the end.
function mul_with_quicksum(f::Signomial{T}, g::Signomial{T}) where T
    n_f = length(f.exp)
    n_g = length(g.exp)
    max_terms = n_f * n_g

    result_coeff = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}()
    sizehint!(result_coeff, max_terms)
    result_exp = Vector{Vector{T}}()
    sizehint!(result_exp, max_terms)

    # Compute all products, allowing duplicates
    @inbounds for i in 1:n_g
        g_exp_i = g.exp[i]
        g_coeff_i = g.coeff[g_exp_i]
        for j in 1:n_f
            f_exp_j = f.exp[j]
            f_coeff_j = f.coeff[f_exp_j]

            # Compute new exponent
            new_exp = g_exp_i .+ f_exp_j
            new_coeff = g_coeff_i * f_coeff_j

            push!(result_exp, new_exp)
            if haskey(result_coeff, new_exp)
                result_coeff[new_exp] += new_coeff
            else
                result_coeff[new_exp] = new_coeff
            end
        end
    end

    return Signomial(result_coeff, result_exp; sorted=false)
end

# Addition of rational signomials
function Base.:+(f::RationalSignomial{T}, g::RationalSignomial{T}) where T
    num = f.num * g.den + f.den * g.num
    den = f.den * g.den
    return RationalSignomial(num, den)
end

# Addition for rational signomials using quicksum addition on the numerator and denominator
# This is an experimental feature and should be used with caution.
function add_with_quicksum(f::RationalSignomial{T}, g::RationalSignomial{T}) where T
    num = quicksum([mul_with_quicksum(f.num, g.den), mul_with_quicksum(f.den, g.num)])
    den = mul_with_quicksum(f.den, g.den)
    return RationalSignomial(num, den)
end

# Quick multiplication for rational signomials
function mul_with_quicksum(F::Vector{RationalSignomial{T}}) where T
    mul = RationalSignomial_one(nvars(F[1]), F[1])
    for f in F
        mul = mul_with_quicksum(mul, f)
    end
    return mul
end

# Quick product over vector of signomials
function mul_with_quicksum(F::Vector{Signomial{T}}) where T
    mul = Signomial_one(nvars(F[1]), F[1])
    for f in F
        mul = mul_with_quicksum(mul, f)
    end
    return mul
end

# Quick addition for vectors of rational signomials
function quicksum(F::Vector{RationalSignomial{T}}) where T
    denoms = [f.den for f in F]
    den = mul_with_quicksum(denoms)
    summand = Vector{Signomial{T}}()
    sizehint!(summand, length(F))
    for i in Base.eachindex(F)
        push!(summand, mul_with_quicksum([j != i ? denoms[j] : F[i].num for j in Base.eachindex(F)]))
    end
    return RationalSignomial(quicksum(summand), den)
end

# Usual multiplication for rational signomials
function Base.:*(f::RationalSignomial{T}, g::RationalSignomial{T}) where T
    num = f.num * g.num
    den = f.den * g.den
    return RationalSignomial(num, den)
end

# Quick multiplication for rational signomials
function mul_with_quicksum(f::RationalSignomial{T}, g::RationalSignomial{T}) where T
    num = mul_with_quicksum(f.num, g.num)
    den = mul_with_quicksum(f.den, g.den)
    return RationalSignomial(num, den)
end

# Division for rational signomials
function Base.:/(f::RationalSignomial{T}, g::RationalSignomial{T}) where T
    num = f.num*g.den
    den = f.den*g.num
    return RationalSignomial(num, den)
end

# Quick division for rational signomials
function div_with_quicksum(f::RationalSignomial{T}, g::RationalSignomial{T}) where T
    num = mul_with_quicksum(f.num, g.den)
    den = mul_with_quicksum(f.den, g.num)
    return RationalSignomial(num, den)
end

# Scalar multiplication for rational signomials
function Base.:*(a::TropicalSemiringElem, f::RationalSignomial{T}) where T
    return RationalSignomial(a*f.num, f.den)
end

# Exponentiation in the tropical semiring
function Base.:^(a::TropicalSemiringElem{typeof(max)}, b::TropicalSemiringElem{typeof(max)})
    R = tropical_semiring(max)
    return R(Rational(a)*Rational(b))
end

# Exponentiation of element of tropical semiring by rational number.
function Base.:^(a::TropicalSemiringElem{typeof(max)}, b::Rational{T}) where T<:Integer
    R = tropical_semiring(max)
    return R(Rational(a)*b)
end

# Exponentiation of element of tropical semiring by float
function Base.:^(a::TropicalSemiringElem{typeof(max)}, b::Float64)
    R = tropical_semiring(max)
    result_val = Float64(Rational(a)) * b
    return R(rationalize(result_val))
end

# exponentiation of a signomial by a positive rational
function Base.:^(f::Signomial, rat::Float64)
    if rat == 0
        return Signomial_one(nvars(f), f)
    else
        new_f_coeff = Dict()
        new_f_exp = copy(f.exp)
        new_f_exp = rat * new_f_exp
        for (key, elem) in f.coeff
            new_f_coeff[rat*key] = elem^rat
        end
        return Signomial(new_f_coeff, new_f_exp; sorted=true)
    end
end

# exponentiation of a rational signomial by a positive rational
function Base.:^(f::RationalSignomial, rat::Float64)
    if rat == 0
        return RationalSignomial_one(nvars(f), f)
    else
        return RationalSignomial(f.num^rat , f.den^rat)
    end
end

function Base.:^(f::Signomial{T}, int::Int64) where T
    new_f_coeff = Dict()
    new_f_exp::Vector{Vector{T}} = copy(f.exp)
    new_f_exp = int * new_f_exp
    for (key, elem) in f.coeff
        new_f_coeff[int*key] = elem^int
    end
    return Signomial(new_f_coeff, new_f_exp; sorted=true)
end

function Base.:^(f::Signomial, int::Rational{T}) where T<:Integer
    new_f_coeff = Dict()
    new_f_exp = convert(Vector{Vector{Rational{BigInt}}}, f.exp)
    new_f_exp::Vector{Vector{Rational{BigInt}}} = Vector{Rational{BigInt}}.(int * new_f_exp)
    for (key, elem) in f.coeff
        new_f_coeff[int*key] = elem^int
    end
    return Signomial(new_f_coeff, new_f_exp; sorted=true)
end

# exponentiation of a rational signomial by a positive integer
function Base.:^(f::RationalSignomial, int::Int64)
    if int == 0
        return RationalSignomial_one(nvars(f), f)
    else
        return RationalSignomial(f.num^int , f.den^int)
    end
end

# exponentiation of a rational signomial by a positive integer
function Base.:^(f::RationalSignomial, int::Rational{T}) where T<:Integer
    if int == 0
        return RationalSignomial_one(nvars(f), f)
    else
        return RationalSignomial(f.num^int , f.den^int)
    end
end

function Base.:*(a::TropicalSemiringElem, f::Signomial{T}) where T
    new_f_coeff = copy(f.coeff)
    new_f_exp = copy(f.exp)
    for i in eachindex(f)
        new_f_coeff[f.exp[i]] = a*f.coeff[f.exp[i]]
    end
    return Signomial(new_f_coeff, new_f_exp; sorted=true)
end

function Base.:(==)(f::Signomial{T}, g::Signomial{T}) where T
    return f.coeff == g.coeff && f.exp == g.exp
end

####################################################################

############# CODE FOR COMPOSITION ##################################

function comp(f::Signomial{T}, G::Vector{Signomial{T}}) where T
    result = Signomial_zero(nvars(G[1]), f)
    # evaluate monomial-wise
    for (exp, coeff) in f.coeff
        term = Signomial_one(nvars(G[1]), f)
        for i in Base.eachindex(G)
            # multiply each variable in the monomial
            term *= G[i]^exp[i]
        end
        result += coeff * term
    end
    return result
end

function comp(f::Signomial{T}, G::Vector{RationalSignomial{T}}) where T
    @req length(G) == nvars(f) "Incorrect number of variables"
    result = RationalSignomial_zero(nvars(G[1]), G[1])
    for (key, val) in f.coeff
        term = RationalSignomial_one(nvars(G[1]), G[1])
        for i in Base.eachindex(G)
            term *= G[i]^key[i]
        end
        result += val * term
    end
    return result
end

# Quick version of composition
function comp_with_quicksum(f::Signomial{T}, G::Vector{RationalSignomial{T}}) where T
    @req length(G) == nvars(f) "Incorrect number of variables"
    summands = Vector{RationalSignomial{T}}()
    sizehint!(summands, length(f.exp))
    for (key, val) in f.coeff
        term = RationalSignomial_one(nvars(G[1]), G[1])
        for i in Base.eachindex(G)
            term = mul_with_quicksum(term, G[i]^key[i])
        end
        push!(summands, val * term)
    end
    return quicksum(summands)
end

function comp(f::RationalSignomial{T}, G::Vector{RationalSignomial{T}}) where T
    num = comp(f.num, G)
    den = comp(f.den, G)
    return num / den
end

# Quick version of composition
function comp_with_quicksum(f::RationalSignomial{T}, G::Vector{RationalSignomial{T}}) where T
    num = comp_with_quicksum(f.num, G)
    den = comp_with_quicksum(f.den, G)
    return div_with_quicksum(num, den)
end

function comp(F::Vector{RationalSignomial{T}}, G::Vector{RationalSignomial{T}}) where T
    return [comp(f, G) for f in F]
end

# Quick version of composition
function comp_with_quicksum(F::Vector{RationalSignomial{T}}, G::Vector{RationalSignomial{T}}) where T
    return [comp_with_quicksum(f, G) for f in F]
end

#########################################################################################################
# Helper functions

# remove all zero monomials from the expression of f
function dedup_monomials(f::Signomial{T}) where T
    new_exp::Vector{Vector{T}} = []
    new_coeff=Dict()
    tropical_zero = zero(f.coeff[f.exp[1]])
    for i in f.exp
        if f.coeff[i] != tropical_zero
            push!(new_exp, i)
            new_coeff[i] = f.coeff[i]
        end
    end
    return Signomial(new_coeff, new_exp)
end

function dedup_monomials(f::RationalSignomial{T}) where T
    return RationalSignomial(dedup_monomials(f.num), dedup_monomials(f.den))
end

function dedup_monomials(F::Vector{RationalSignomial{T}}) where T
    return [dedup_monomials(f) for f in F]
end

# Count the number of monomials appearing in a tropical expression
function monomial_count(f::Signomial{T}) where T
    return length(f.exp)
end

# Count the number of monomials appearing in a tropical expression
function monomial_count(f::RationalSignomial{T}) where T
    return monomial_count(f.num) + monomial_count(f.den)
end

####################### PRETTY PRINTING #######################################

# Unicode subscript digits for variable names like x₁, x₂, ...
const _SUBSCRIPT_DIGITS = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉']

function _subscript(n::Integer)
    return join(_SUBSCRIPT_DIGITS[d+1] for d in reverse(digits(n)))
end

# Format a scalar exponent value as a string, dispatching on its type.
_exp_str(α::AbstractFloat) = isinteger(α) ? "$(Int(α))" : "$(α)"
_exp_str(α::Rational)      = denominator(α) == 1 ? "$(numerator(α))" : "$(α)"
_exp_str(α::Integer)       = "$(α)"
_exp_str(α)                = "$(α)"  # fallback

# Format a scalar coefficient (underlying value from the tropical semiring).
_coeff_str(c::AbstractFloat) = isinteger(c) ? "$(Int(c))" : "$(c)"
_coeff_str(c::Rational)      = denominator(c) == 1 ? "$(numerator(c))" : "$(c)"
_coeff_str(c::Integer)       = "$(c)"
_coeff_str(c)                = "$(c)"  # fallback

# Format a single monomial as a human-readable string.
# A monomial is c ⊙ x₁^α₁ ⊙ x₂^α₂ ⊙ ... which in classical terms means
# c + α₁x₁ + α₂x₂ + ...
function _monomial_str(coeff::Oscar.TropicalSemiringElem, exp::Vector{T}) where T
    c = Oscar.data(coeff)   # underlying Rational (or Int) value
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

    # prepend the constant coefficient if non-zero
    if c != 0
        pushfirst!(terms, _coeff_str(c))
    end
    return join(terms, " + ")
end

function Base.show(io::IO, f::Signomial{T}) where T
    if isempty(f.exp)
        print(io, "max()")
        return
    end
    strs = [_monomial_str(f.coeff[e], e) for e in f.exp]
    print(io, "max(", join(strs, ", "), ")")
end

function Base.show(io::IO, f::RationalSignomial{T}) where T
    print(io, "(", f.num, ") ⊘ (", f.den, ")")
end

function Base.show(io::IO, F::Vector{RationalSignomial{T}}) where T
    for (i, f) in enumerate(F)
        print(io, "f$(_subscript(i)) = ", f, "\n")
    end
end

# Callable interface: f(x) as syntactic sugar for evaluate(f, x)
(f::Signomial)(x::Vector) = evaluate(f, x)
(f::RationalSignomial)(x::Vector) = evaluate(f, x)

# Count the number of monomials appearing in a tropical expression
function monomial_count(F::Vector{RationalSignomial{T}}) where T
    return sum([monomial_count(f) for f in F])
end
