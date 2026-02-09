# This file contains functions to convert a multilayer perceptron to a tropical Puiseux rational function, and to remove redundant monomials from the resulting function.

@doc raw"""
    monomial_strong_elim(f::TropicalPuiseuxPoly{T}; parallel::Bool=true)

Removes redundant monomials from a tropical Puiseux polynomial f.

A monomial is considered redundant if its corresponding polyhedron (the region
where that monomial dominates) is not full-dimensional.

When `parallel=true` (default), uses multithreading to parallelize the
full-dimensionality checks, which can provide significant speedup for
polynomials with many monomials.

# Arguments
- `f::TropicalPuiseuxPoly{T}`: The polynomial to simplify
- `parallel::Bool=true`: Whether to use parallel computation

# Returns
- `TropicalPuiseuxPoly{T}`: A new polynomial with redundant monomials removed

# Note
The parallel version requires Julia to be started with multiple threads
(e.g., `julia -t auto` or `JULIA_NUM_THREADS=4 julia`).
"""
function monomial_strong_elim(f::TropicalPuiseuxPoly{T}; parallel::Bool=true) where T
    n = length(f.exp)

    if parallel && Threads.nthreads() > 1 && n > 1
        # Parallel version: check full-dimensionality for all monomials in parallel
        keep = Vector{Bool}(undef, n)

        Threads.@threads for i in 1:n
            poly = polyhedron(f, i)
            keep[i] = Oscar.is_fulldimensional(poly)
        end

        # Collect results based on keep vector
        new_exp = Vector{Vector{T}}()
        sizehint!(new_exp, count(keep))
        new_coeff = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}()

        for i in 1:n
            if keep[i]
                e = f.exp[i]
                push!(new_exp, e)
                new_coeff[e] = f.coeff[e]
            end
        end

        return TropicalPuiseuxPoly(new_coeff, new_exp)
    else
        # Sequential version (original algorithm)
        new_exp = Vector{Vector{T}}()
        sizehint!(new_exp, n)
        new_coeff = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}()

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
end 

@doc raw"""
    monomial_strong_elim(f::TropicalPuiseuxRational{T}; parallel::Bool=true)

Removes redundant monomials from both numerator and denominator of a tropical
Puiseux rational function.

# Arguments
- `f::TropicalPuiseuxRational{T}`: The rational function to simplify
- `parallel::Bool=true`: Whether to use parallel computation
"""
function monomial_strong_elim(f::TropicalPuiseuxRational{T}; parallel::Bool=true) where T
    return TropicalPuiseuxRational(
        monomial_strong_elim(f.num; parallel=parallel),
        monomial_strong_elim(f.den; parallel=parallel)
    )
end

@doc raw"""
    monomial_strong_elim(F::Vector{TropicalPuiseuxRational{T}}; parallel::Bool=true)

Removes redundant monomials from a vector of tropical Puiseux rational functions.

# Arguments
- `F::Vector{TropicalPuiseuxRational{T}}`: The vector of rational functions to simplify
- `parallel::Bool=true`: Whether to use parallel computation
"""
function monomial_strong_elim(F::Vector{TropicalPuiseuxRational{T}}; parallel::Bool=true) where T
    return [monomial_strong_elim(f; parallel=parallel) for f in F]
end

"""
    mlp_to_trop_with_strong_elim(linear_maps, bias, thresholds)

**DEPRECATED**: Use `mlp_to_trop(linear_maps, bias, thresholds, strong_elim=true, dedup=true)` instead.

Computes the tropical Puiseux rational function associated to a multilayer perceptron,
and runs monomial_strong_elim at each layer to remove redundant monomials.
"""
function mlp_to_trop_with_strong_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    @warn "mlp_to_trop_with_strong_elim is deprecated, use mlp_to_trop(..., strong_elim=true, dedup=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, strong_elim=true, dedup=true)
end 

"""
    mlp_to_trop_with_quicksum_with_strong_elim(linear_maps, bias, thresholds)

**DEPRECATED**: Use `mlp_to_trop(linear_maps, bias, thresholds, quicksum=true, strong_elim=true)` instead.

Computes the tropical Puiseux rational function associated to a multilayer perceptron.
Runs monomial_strong_elim at each layer, and uses quicksum operations for tropical objects.
"""
function mlp_to_trop_with_quicksum_with_strong_elim(linear_maps::Vector{Matrix{T}}, bias, thresholds) where T<:Union{Oscar.scalar_types, Rational{BigInt}}
    @warn "mlp_to_trop_with_quicksum_with_strong_elim is deprecated, use mlp_to_trop(..., quicksum=true, strong_elim=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, quicksum=true, strong_elim=true)
end