# This file contains functions to convert a multilayer perceptron to a tropical Puiseux rational function, and to remove redundant monomials from the resulting function.

@doc raw"""
    monomial_strong_elim(f::TropicalPuiseuxPoly{T}) removes redundant monomials from a tropical Puiseux polynomial f.
    
    inputs: f: an object of type TropicalPuiseuxPolynomial.
    outputs: an object of type TropicalPuiseuxPolynomial.
"""
function monomial_strong_elim(f::TropicalPuiseuxPoly{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    new_exp = Vector{Vector{T}}()
    sizehint!(new_exp, length(f.exp))
    new_coeff = Dict()
    # iterate through the monomials and removes the redundant ones, i.e. the ones 
    # whose corresponding polyhedron is not full-dimensional
    for i in Base.eachindex(f.exp)
        poly = polyhedron(f, i)
        if Oscar.is_fulldimensional(poly)
            e = f.exp[i] 
            push!(new_exp, e)
            new_coeff[e] = f.coeff[e]
        end 
    end 
    return TropicalPuiseuxPoly(new_coeff, new_exp)
end 

@doc raw"""
    monomial_strong_elim(f::TropicalPuiseuxRational{T}) removes redundant monomials from a tropical Puiseux rational function f.
    
    inputs: f: an object of type TropicalPuiseuxRational.
    outputs: an object of type TropicalPuiseuxRational.
"""
function monomial_strong_elim(f::TropicalPuiseuxRational{T}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return TropicalPuiseuxRational(monomial_strong_elim(f.num), monomial_strong_elim(f.den))
end

@doc raw"""
    monomial_strong_elim(f::Vector{TropicalPuiseuxRational{T}}) removes redundant monomials from a vector of tropical Puiseux rational functions F.
    
    inputs: f: an object of type TropicalPuiseuxRational.
    outputs: an object of type TropicalPuiseuxRational.
"""
function monomial_strong_elim(F::Vector{TropicalPuiseuxRational{T}}) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    return [monomial_strong_elim(f) for f in F]
end

@doc raw"""
    mlp_to_trop_with_strong_elim(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron, and runs monomial_strong_elim at each layer to remove redundant monomials.

    inputs: linear maps: an array containing the weight matrices of the neural network. 
            bias: an array containing the biases at each layer
            thresholds: an array containing the threshold of the activation function at each layer, i.e. the number t such that the activation is of
            the form x => max(x,t).
    outputs: an object of type TropicalPuiseuxRational.
"""
function mlp_to_trop_with_strong_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    R = tropical_semiring(max)
    # initialisation: the first vector of tropical rational functions is just the identity function
    output = single_to_trop(linear_maps[1], bias[1], thresholds[1])
    output = dedup_monomials(output)
    # iterate through the layers and compose variable output with the current layer at each step
    for i in Base.eachindex(linear_maps)
        A = linear_maps[i]
        b = bias[i]
        t = thresholds[i]
        #check sizes agree
        if size(A, 1) != length(b) || size(A, 1) != length(t) 
            # stricly speaking this should be implemented as an exception
            println("Dimensions of matrix don't agree with constant term or threshold")
        end 
        if i != 1
            # compute the vector of tropical rational functions corresponding to the function 
            # max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
            ith_tropical = single_to_trop(A, b, t)
            # compose this with the output of the previous layer
            output = comp(ith_tropical, output)
            output = monomial_strong_elim(output)
            #output = dedup_monomials(output)
        end 
    end 
    return output
end 

@doc raw"""
    mlp_to_trop_with_quicksum_with_strong_elim(linear_maps, bias, thresholds) computes the tropical Puiseux rational function associated to a multilayer perceptron. Runs monomial_strong_elim at each layer, and uses quicksum operations for tropical objects.
    
    inputs: linear maps: an array containing the weight matrices of the neural network. 
            bias: an array containing the biases at each layer
            thresholds: an array containing the threshold of the activation function at each layer, i.e. the number t such that the activation is of
            the form x => max(x,t).
    outputs: an object of type TropicalPuiseuxRational.
"""
function mlp_to_trop_with_quicksum_with_strong_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
        R = tropical_semiring(max)
        # initialisation: the first vector of tropical rational functions is just the identity function
        output = single_to_trop(linear_maps[1], bias[1], thresholds[1])
        # iterate through the layers and compose variable output with the current layer at each step
        for i in Base.eachindex(linear_maps)
            A = linear_maps[i]
            b = bias[i]
            t = thresholds[i]
            #check sizes agree
            if size(A, 1) != length(b) || size(A, 1) != length(t) 
                # stricly speaking this should be implemented as an exception
            end 
            if i != 1
                # compute the vector of tropical rational functions corresponding to the function 
                # x => max(Ax+b, t) where A = linear_maps[i], b = bias[i] and t = thresholds[i]
                ith_tropical = single_to_trop(A, b, t)
                # compose this with the output of the previous layer
                output = comp_with_quicksum(ith_tropical, output)
                output = monomial_strong_elim(output)
            end 
        end 
    return output
end