####################### BASIC CONSTRUCTIONS ##################################

"""
Represents a tropical Puiseux polynomial, i.e. a tropical polynomial in several variables, whose 
exponents might be rational numbers (i.e. we use this structure when T is a subtype of the rational numbers).
The coefficients are elements of the tropical semiring.

# Example 

julia> f = TropicalPuiseuxPoly(Dict([1, 2] => 1, [2, 1] => 2), [[1, 2], [2, 1]])
  TropicalPuiseuxPoly{Int64}(Dict([2, 1] => 2, [1, 2] => 1), [[1, 2], [2, 1]])

"""
struct TropicalPuiseuxPoly{T}
    coeff::Dict
    exp::Vector{Vector{T}}
end 

"""
Represents a quotient of tropical Puiseux polynomials.
"""
struct TropicalPuiseuxRational{T}
    num::TropicalPuiseuxPoly{T}
    den::TropicalPuiseuxPoly{T}
end 

@doc raw"""
    TropicalPuiseuxPoly(coeff::Dict, exp::Vector{Vector{T}}, sorted::Bool)

Constructs a tropical Puiseux polynomial from a dictionary of coefficients and a vector of exponents, by first sorting the exponents lexicographically.
"""
function TropicalPuiseuxPoly(coeff::Dict, exp::Vector{Vector{T}}, sorted) where T
    # first we need to order everything lexicographically
    if !sorted 
        exp = sort(exp)
    end 
    return TropicalPuiseuxPoly(coeff, exp)
end 

@doc raw"""
    TropicalPuiseuxPoly(coeff::Dict, exp::Vector{Vector{T}}, sorted::Bool)

Constructs a tropical Puiseux polynomial from a vector of coefficients and a vector of exponents, by first sorting the exponents lexicographically, and then constructing the dictionary of coefficients.

```jldoctest
julia> f = TropicalPuiseuxPoly([1, 2], [[1, 2], [2, 1]])
  TropicalPuiseuxPoly{Int64}(Dict([2, 1] => 2, [1, 2] => 1), [[1, 2], [2, 1]])
```
"""
function TropicalPuiseuxPoly(coeff::Vector, exp::Vector, sorted)
    if !sorted 
        I = sortperm(exp)
        exp = exp[I]
        coeff = coeff[I]
    end 
    return TropicalPuiseuxPoly(Dict(zip(exp, coeff)), exp)
end 

@doc raw"""
    TropicalPuiseuxPoly_const(n, c, f::TropicalPuiseuxPoly{T})

Outputs the constant c viewed as a tropical Puiseux polynomial in n variables, and exponents in the 
same type as f.
"""
function TropicalPuiseuxPoly_const(n, c, f::TropicalPuiseuxPoly{T}) where T
    exp = [Base.zeros(T, n)]
    coeff = Dict(Base.zeros(T, n) => c)
    return TropicalPuiseuxPoly(coeff, exp)
end 

@doc raw"""
    TropicalPuiseuxPoly_zero(n, f::TropicalPuiseuxPoly{T})
Ouputs the tropical zero viewed as a tropical Puiseux polynomial in n variables, and exponents in the
same type as f.
"""
function TropicalPuiseuxPoly_zero(n::Int64, f::TropicalPuiseuxPoly{T}) where T
    return TropicalPuiseuxPoly_const(n, zero(f.coeff[f.exp[1]]), f)
end 

@doc raw"""
    TropicalPuiseuxPoly_one(n, f::TropicalPuiseuxPoly{T})
Ouputs the tropical one viewed as a tropical Puiseux polynomial in n variables, and exponents in the
same type as f.
"""
function TropicalPuiseuxPoly_one(n::Int64, f::TropicalPuiseuxPoly{T}) where T
    return TropicalPuiseuxPoly_one(n, one(f.coeff[f.exp[1]]), f)
end 

# This is deprecated and should be removed in the future
function TropicalPuiseuxPoly_one(n::Int64, c::TropicalSemiringElem, f::TropicalPuiseuxPoly{T}) where T
    return TropicalPuiseuxPoly_const(n, one(c), f)
end

@doc raw"""
    TropicalPuiseuxMonomial(c, exp::Vector{T}) 

Constructs a tropical Puiseux polynomial from a scalar c and a vector of exponents. This is a monomial whose 
coefficient is c and exponents are given by exp.
"""
function TropicalPuiseuxMonomial(c, exp::Vector{T}) where T
    return TropicalPuiseuxPoly([c for _ in 1:length(exp)], [exp], true)
end

@doc raw"""
    TropicalPuiseuxRational(f)

Constructs a tropical Puiseux rational function from a tropical Puiseux polynomial f, 
by setting the denominator to be the tropical one.
"""
function TropicalPuiseuxPoly_to_rational(f)
    return TropicalPuiseuxRational(f, TropicalPuiseuxPoly_one(nvars(f), f))
end 

function TropicalPuiseuxPoly_zero(n, f::TropicalPuiseuxPoly{T}) where T
    exp = [Base.zeros(T, n)]
    coeff = Dict(Base.zeros(n) => zero(f.coeff[f.exp[1]]))
    return TropicalPuiseuxPoly(coeff, exp)
end 

@doc raw"""
The identity function viewed as a tropical Puiseux rational function in n variables.
"""
function TropicalPuiseuxRational_identity(n, c)
    output = Vector{TropicalPuiseuxRational}()
    sizehint!(output, n)
    for i in 1:n 
        # add the i-th coordinate viewed as a tropical rational function 
        push!(output, TropicalPuiseuxPoly_to_rational( 
            TropicalPuiseuxMonomial(one(c), [j == i ? 1 : 0 for j in 1:n])))
    end 
    return output
end 

@doc raw"""
Returns an iterator for the exponents of a tropical Puiseux polynomial.
"""
function eachindex(f::TropicalPuiseuxPoly)
    return Base.eachindex(f.exp)
end 

@doc raw"""
Returns the number of variables of a tropical Puiseux polynomial.
"""
function Oscar.nvars(f::TropicalPuiseuxPoly)
    if !is_empty(f.coeff)
        return length(f.exp[1])
    else 
        return -1
    end 
end 

@doc raw"""
Returns the number of variables of a tropical Puiseux rational.
"""
function Oscar.nvars(f::TropicalPuiseuxRational)
    return Oscar.nvars(f.den)
end 

@doc raw"""
Outputs zero, viewed as a tropical Puiseux rational function in n variables, and with exponents in the same type as f.
"""
function TropicalPuiseuxRational_zero(n::Int64, f::TropicalPuiseuxRational{T}) where T
    return TropicalPuiseuxRational(TropicalPuiseuxPoly_zero(n, f.num), TropicalPuiseuxPoly_one(n, f.den))
end 

@doc raw"""
Outputs one, viewed as a tropical Puiseux rational function in n variables, and with exponents in the same type as f.
"""
function TropicalPuiseuxRational_one(n::Int64, f)
    return TropicalPuiseuxRational(TropicalPuiseuxPoly_one(n, f.num), TropicalPuiseuxPoly_one(n, f.num))
end 

##################################################################################

####################### STRING REPRESENTATIONS ###################################

function Base.string(f::TropicalPuiseuxPoly{T}) where T
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

function Base.repr(f::TropicalPuiseuxPoly)
    return string(f)
end 

function Base.string(f::TropicalPuiseuxRational)
    return string(f.num) * " / " * string(f.den)
end 

######################################################################

########################## EVALUATION ################################

#### This section defines API to evaluate Tropical Puiseux Polynomials ####

@doc raw"""
    eval(f::TropicalPuiseuxPoly, a::Vector)
Evaluates the tropical Puiseux polynomial f at the point a.
"""
function eval(f::TropicalPuiseuxPoly{T}, a::Vector) where T
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
    eval(f::TropicalPuiseuxRational{T}, a::Vector)
Evaluates the tropical Puiseux rational function f at the point a.
"""
function eval(f::TropicalPuiseuxRational{T}, a::Vector) where T
    n::TropicalSemiringElem{typeof(max)} = eval(f.num, a) 
    m::TropicalSemiringElem{typeof(max)} = eval(f.den, a) 
    return n / m
end 

@doc raw"""
    eval(F::Vector{TropicalPuiseuxPoly{T}}, a::Vector)
Evaluates the vector of tropical Puiseux rationals F at the point a.
"""
function eval(F::Vector{TropicalPuiseuxRational{T}}, a::Vector) where T
    return [eval(f, a) for f in F]
end

######################################################################

################ ARITHMETIC OPERATIONS ###############################

#### This section implements standard arithmetic operations for tropical 
#### polynomials and rational functions                                  

function Base.:/(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T
    return TropicalPuiseuxRational(f, g)
end 

# Quicker version of addition for vectors of tropical polynomials
# *Warning*: this doesn't sort the exponents of the resulting polynomial, so should never be used if the 
# output is to be used in further computations that require the exponents to be sorted.
function quicksum(F::Vector{TropicalPuiseuxPoly{T}}) where T
    R = tropical_semiring(max)
    terms = reduce(vcat, [f.exp for f in F])
    h_exp::Vector{Vector{T}} = terms
    #terms = unique(flatview(VectorOfArray([f.exp for f in F])))
    #sort!(terms)
    h_coeff = Dict()
    for exp in terms
        h_coeff[exp] = sum([f.coeff[exp] for f in F if haskey(f.coeff, exp)])
    end
    return TropicalPuiseuxPoly(h_coeff, h_exp, true)
end

"""
Takes two TropicalPuiseuxPoly whose exponents are lexicographically ordered and outputs the sum with 
lexicoraphically ordered exponents
"""
function Base.:+(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T
    lf = length(f.coeff)
    lg = length(g.coeff)
    # initialise coeffs and exponents vectors for the sum h = f + g
    h_coeff = Dict()
    h_exp = Vector{Vector{T}}()
    # initialise indexing variable for loops below
    j=1
    # at each term of g, check if there is a term of f with matching exponents
    for i in eachindex(g)
        c = g.exp[i] 
        added = false
        # loop through terms of f ordered lexicographically, until we reach a term with a larger power 
        while j <= lf
            d = f.exp[j]
            if d > c 
                if g.coeff[c] != zero(g.coeff[c])
                    # if c > d then we have reached the first exponent of f larger than c so we can stop here, and add the i-th 
                    # term of g to h.
                    h_coeff[c] = g.coeff[c]
                    push!(h_exp, c)
                    added = true
                end 
                break
            elseif c == d 
                if g.coeff[c] != zero(g.coeff[c]) || f.coeff[c] != zero(g.coeff[c])
                    # if we reach an equal exponent, both get added simultaneously to the sum.
                    h_coeff[c] = f.coeff[c]+g.coeff[c]
                    push!(h_exp, c)
                    added = true
                end 
                # update j for the iteration with the next i
                j+=1
                break
            else 
                if f.coeff[d] != zero(f.coeff[d])
                    # if d < c then we can add that exponent of f to the sum
                    h_coeff[d] = f.coeff[d]
                    push!(h_exp, d)
                end 
                j+=1
            end 
            # Note about the indexing variable j:
            # Since a iteration i, we stop when we have either reached a j whose corresponding exponent is too large, or 
            # equal to that of i, we can start the iteration of i+1 at the j at which the previous iteration stopped.
        end 
        # if we have exchausted all terms of f then we need to add all the remaining terms of g
        if !added && j > lf && g.coeff[c] != zero(g.coeff[c])
            h_coeff[c] = g.coeff[c]
            push!(h_exp, c)
        end 
    end 
    # once we have exhausted all terms of g, we need to check for remaining terms of f
    while j <= lf 
        d = f.exp[j]
        if f.coeff[d] != zero(f.coeff[d])
            h_coeff[d] = f.coeff[d]
            push!(h_exp, d)
        end 
        j+=1
    end 
    h = TropicalPuiseuxPoly(h_coeff, h_exp, true)
    return h
end 

"""
Takes two TropicalPuiseuxPoly whose exponents are lexicographically ordered and outputs the product with 
lexicoraphically ordered exponents
"""
function Base.:*(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T
    prod = TropicalPuiseuxPoly_zero(nvars(f), f)
    # if f = a_0 + ... + a_n T^n and g = b_0 + ... + b_n then the product is 
    # the sum of all the b_i T^i * f
    for i in eachindex(g)
        term_coeff = Dict()
        # compute the coefficients of b_i T^i * f
        for (key, elem) in f.coeff
            term_coeff[key+g.exp[i]] = g.coeff[g.exp[i]] * elem
        end 
        # compute the exponenets of b_i T^i * f
        term_exp = [g.exp[i] + f.exp[j] for j in eachindex(f)]
        prod += TropicalPuiseuxPoly(term_coeff, term_exp, true)
    end 
    return prod 
end 

# Multiplication of tropical polynomials with using quicksum addition.
function mul_with_quicksum(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T
    # if f = a_0 + ... + a_n T^n and g = b_0 + ... + b_n then the product is 
    # the sum of all the b_i T^i * f
    summands = Vector{TropicalPuiseuxPoly{T}}()
    sizehint!(summands, length(g.exp))
    for i in eachindex(g)
        term_coeff = Dict()
        # compute the coefficients of b_i T^i * f
        for (key, elem) in f.coeff
            #println(nvars(f), " ", nvars(g))
            term_coeff[key+g.exp[i]] = g.coeff[g.exp[i]] * elem
        end 
        # compute the exponenets of b_i T^i * f
        term_exp = [g.exp[i] + f.exp[j] for j in eachindex(f)]
        push!(summands, TropicalPuiseuxPoly(term_coeff, term_exp, true))
    end 
    return quicksum(summands)
end 

# Addition of tropical Puiseux rationals
function Base.:+(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T
    num = f.num * g.den + f.den * g.num 
    den = f.den * g.den 
    return TropicalPuiseuxRational(num, den) 
end 

# Addition for tropical Puiseux rationals using quicksum addition on the numerator and denominator
# This is an experimental feature and should be used with caution.
function add_with_quicksum(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T
    num = quicksum([mul_with_quicksum(f.num, g.den), mul_with_quicksum(f.den, g.num)])
    den = mul_with_quicksum(f.den, g.den) 
    return TropicalPuiseuxRational(num, den) 
end 

# Quick multiplication for rational functions
function mul_with_quicksum(F::Vector{TropicalPuiseuxRational{T}}) where T
    mul = TropicalPuiseuxPuiseux_one(nvars(f), f)
    for f in F 
        mul = mul_with_quicksum(mul, f)
    end
    return mul
end 

# Quick product over vector of polynomials
function mul_with_quicksum(F::Vector{TropicalPuiseuxPoly{T}}) where T
    mul = TropicalPuiseuxPoly_one(nvars(F[1]), F[1])
    for f in F 
        mul = mul_with_quicksum(mul, f)
    end 
    return mul
end 

# Quick addition for vectors of rational functions
function quicksum(F::Vector{TropicalPuiseuxRational{T}}) where T
    denoms = [f.den for f in F]
    den = mul_with_quicksum(denoms)
    summand = Vector{TropicalPuiseuxPoly{T}}()
    sizehint!(summand, length(F))
    for i in Base.eachindex(F)
        push!(summand, mul_with_quicksum([j != i ? denoms[j] : F[i].num for j in Base.eachindex(F)]))
    end
    return TropicalPuiseuxRational(quicksum(summand), den) 
end 

# Usual multiplication for rational functions
function Base.:*(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T
    num = f.num * g.num 
    den = f.den * g.den 
    return TropicalPuiseuxRational(num, den)
end 

# Quick multiplication for rational functions
function mul_with_quicksum(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T
    num = mul_with_quicksum(f.num, g.num) 
    den = mul_with_quicksum(f.den, g.den) 
    return TropicalPuiseuxRational(num, den)
end 

# Division for vectors of rational functions
function Base.:/(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T
    num = f.num*g.den 
    den = f.den*g.num
    return TropicalPuiseuxRational(num, den)
end 

# Quick division for vectors of rational functions
function div_with_quicksum(f::TropicalPuiseuxRational{T}, g::TropicalPuiseuxRational{T}) where T
    num = mul_with_quicksum(f.num, g.den) 
    den = mul_with_quicksum(f.den, g.num)
    return TropicalPuiseuxRational(num, den)
end 

# Scalar multiplication for rational functions
function Base.:*(a::TropicalSemiringElem, f::TropicalPuiseuxRational{T}) where T
    return TropicalPuiseuxRational(a*f.num, f.den)
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

# exponentiation of a tropical Puiseux polynomial by a positive rational
function Base.:^(f::TropicalPuiseuxPoly, rat::Float64)
    if rat == 0
        return TropicalPuiseuxPoly_one(nvars(f), f)
    else 
        new_f_coeff = Dict()
        new_f_exp = copy(f.exp)
        new_f_exp = rat * new_f_exp 
        for (key, elem) in f.coeff
            new_f_coeff[rat*key] = elem^rat
        end 
        return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
    end 
end 

# exponentiation of a tropical Puiseux rational function by a positive rational
function Base.:^(f::TropicalPuiseuxRational, rat::Float64)
    if rat == 0
        return TropicalPuiseuxRational_one(nvars(f), f)
    else 
        return TropicalPuiseuxRational(f.num^rat , f.den^rat)
    end 
end 

function Base.:^(f::TropicalPuiseuxPoly{T}, int::Int64) where T
    new_f_coeff = Dict()
    new_f_exp::Vector{Vector{T}} = copy(f.exp)
    new_f_exp = int * new_f_exp 
    for (key, elem) in f.coeff
        new_f_coeff[int*key] = elem^int
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end 

function Base.:^(f::TropicalPuiseuxPoly, int::Rational{T}) where T<:Integer
    new_f_coeff = Dict()
    new_f_exp = convert(Vector{Vector{Rational{BigInt}}}, f.exp)
    new_f_exp::Vector{Vector{Rational{BigInt}}} = Vector{Rational{BigInt}}.(int * new_f_exp) 
    for (key, elem) in f.coeff
        new_f_coeff[int*key] = elem^int
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end 

# exponentiation of a tropical Puiseux rational function by a positive integer
function Base.:^(f::TropicalPuiseuxRational, int::Int64)
    if int == 0
        return TropicalPuiseuxRational_one(nvars(f), f)
    else 
        return TropicalPuiseuxRational(f.num^int , f.den^int)
    end 
end 

# exponentiation of a tropical Puiseux rational function by a positive integer
function Base.:^(f::TropicalPuiseuxRational, int::Rational{T}) where T<:Integer
    if int == 0
        return TropicalPuiseuxRational_one(nvars(f), f)
    else 
        return TropicalPuiseuxRational(f.num^int , f.den^int)
    end 
end 

function Base.:*(a::TropicalSemiringElem, f::TropicalPuiseuxPoly{T}) where T
    new_f_coeff = copy(f.coeff)
    new_f_exp = copy(f.exp)
    for i in eachindex(f)
        new_f_coeff[f.exp[i]] = a*f.coeff[f.exp[i]]
    end 
    return TropicalPuiseuxPoly(new_f_coeff, new_f_exp, true)
end

function Base.:(==)(f::TropicalPuiseuxPoly{T}, g::TropicalPuiseuxPoly{T}) where T
    return f.coeff == g.coeff && f.exp == g.exp
end

####################################################################

############# CODE FOR COMPOSITION ##################################

function comp(f::TropicalPuiseuxPoly{T}, G::Vector{TropicalPuiseuxPoly{T}}) where T
    comp = TropicalPuiseuxPoly_zero(nvars(G[1]), f)
    # evaluate monomial-wise
    for (exp, coeff) in f.coeff
        term = TropicalPuiseuxPoly_one(nvars(G[1]), f)
        for i in Base.eachindex(G)
            # multiply each variable in the monomial 
            term *= G[i]^exp[i]
        end 
        comp += coeff * term
    end 
    return comp
end

function comp(f::TropicalPuiseuxPoly{T}, G::Vector{TropicalPuiseuxRational{T}}) where T
    @req length(G) == nvars(f) "Incorrect number of variables"
    comp = TropicalPuiseuxRational_zero(nvars(G[1]), G[1])
    for (key, val) in f.coeff
        term = TropicalPuiseuxRational_one(nvars(G[1]), G[1])
        for i in Base.eachindex(G)
            term *= G[i]^key[i]
        end
        comp += val * term 
    end 
    return comp
end

# Quick version of composition 
function comp_with_quicksum(f::TropicalPuiseuxPoly{T}, G::Vector{TropicalPuiseuxRational{T}}) where T
    @req length(G) == nvars(f) "Incorrect number of variables"
    summands = Vector{TropicalPuiseuxRational{T}}()
    sizehint!(summands, length(f.exp))
    comp = TropicalPuiseuxRational_zero(nvars(G[1]), G[1])
    for (key, val) in f.coeff
        term = TropicalPuiseuxRational_one(nvars(G[1]), G[1])
        for i in Base.eachindex(G)
            term = mul_with_quicksum(term, G[i]^key[i])
        end
        push!(summands, val * term)
    end 
    return quicksum(summands)
end

function comp(f::TropicalPuiseuxRational{T}, G::Vector{TropicalPuiseuxRational{T}}) where T
    num =  comp(f.num, G)
    den = comp(f.den, G)
    val = num / den 
    return val 
end 

# Quick version of composition 
function comp_with_quicksum(f::TropicalPuiseuxRational{T}, G::Vector{TropicalPuiseuxRational{T}}) where T
    num =  comp_with_quicksum(f.num, G)
    den = comp_with_quicksum(f.den, G)
    val = div_with_quicksum(num, den) 
    return val 
end 

function comp(F::Vector{TropicalPuiseuxRational{T}}, G::Vector{TropicalPuiseuxRational{T}}) where T
    return [comp(f, G) for f in F]
end 

# Quick version of composition 
function comp_with_quicksum(F::Vector{TropicalPuiseuxRational{T}}, G::Vector{TropicalPuiseuxRational{T}}) where T
    return [comp_with_quicksum(f, G) for f in F]
end

#########################################################################################################
# Helper functions 

# remove all zero monomials from the expression of f
function dedup_monomials(f::TropicalPuiseuxPoly{T}) where T
    new_exp::Vector{Vector{T}} = []
    new_coeff=Dict()
    tropical_zero = zero(f.coeff[f.exp[1]])
    for i in f.exp
        if i != tropical_zero
            push!(new_exp, i)
            new_coeff[i] = f.coeff[i]
        else 
            println("found a zero")
        end
    end 
    return TropicalPuiseuxPoly(new_coeff, new_exp)
end 

function dedup_monomials(f::TropicalPuiseuxRational{T}) where T
    return TropicalPuiseuxRational(dedup_monomials(f.num), dedup_monomials(f.den))
end 

function dedup_monomials(F::Vector{TropicalPuiseuxRational{T}}) where T
    return [dedup_monomials(f) for f in F]
end 

# Count the number of monomials appearing in a tropical expression
function monomial_count(f::TropicalPuiseuxPoly{T}) where T
    return length(f.exp)
end 

# Count the number of monomials appearing in a tropical expression
function monomial_count(f::TropicalPuiseuxRational{T}) where T
    return monomial_count(f.num) + monomial_count(f.den)
end 

# Count the number of monomials appearing in a tropical expression
function monomial_count(F::Vector{TropicalPuiseuxRational{T}}) where T
    return sum([monomial_count(f) for f in F])
end