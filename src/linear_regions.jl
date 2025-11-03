# This file contains code to compute 
# the linear regions of tropical Puiseux polynomials and rational functions.

@doc raw"""
    polyhedron(f::TropicalPuiseuxPoly, i::Int)

Ouputs the polyhedron corresponding to points where f is given by the 
linear map corresponding to the i-th monomial of f. 

# Example
Output the polyhedron where f = max(x, y) is equal to x
```jldoctest
julia> f = TropicalPuiseuxPoly(Dict([1, 0] => 0, [0, 1] => 0), [[1, 0], [0, 1]]);

julia> polyhedron(f, [1, 0])
Polyhedron in ambient dimension 2 with Float64 type coefficients
```
"""
function polyhedron(f::TropicalPuiseuxPoly, i)
    # take A to be the matrix with rows αⱼ - αᵢ for all j ≠ i, where the αᵢ are the exponents of f.
    A = mapreduce(permutedims, vcat, [Float64.(f.exp[j]) - Float64.(f.exp[i]) for j in eachindex(f)])
    # and b the vector whose j-th entry is f.coeff[αⱼ] - f.coeff[αᵢ] for all j ≠ i. 
    b = [Float64(Rational(f.coeff[f.exp[i]])) - Float64(Rational(f.coeff[j])) for j in f.exp]
    # The polyhedron is then the set of points x such that Ax ≤ b.
    return Oscar.polyhedron(A, b)
end

@doc raw"""
    enum_linear_regions(f::TropicalPuiseuxPoly) 
    
Outputs an array of tuples (poly, bool) indexed by the same set as the exponents of f. The tuple element poly 
is the linear region corresponding to the exponent, and bool is true when this region is nonemtpy.

# Example
Enumerates the linear regions of f = max(x, y).
```jldoctest
julia> f = TropicalPuiseuxPoly(Dict([1, 0] => 0, [0, 1] => 0), [[1, 0], [0, 1]]);

julia> enum_linear_regions(f)
2-element Vector{Any}:
 (Polyhedron in ambient dimension 2 with Float64 type coefficients, true)
 (Polyhedron in ambient dimension 2 with Float64 type coefficients, true)
```
"""
function enum_linear_regions(f::TropicalPuiseuxPoly)
    linear_regions = Vector()
    sizehint!(linear_regions, length(f.exp))
    for i in eachindex(f)
        poly = polyhedron(f, i)
        # add the polyhedron to the list plus a bool saying whether the polyhedron is non-empty
        # TODO: this should be replaced by a check that the polyhedron is full dimensional for performance.
        push!(linear_regions, (poly, Oscar.is_feasible(poly)))
    end
    return linear_regions
end

# Computes the number of equivalence classes of the transitive closure of a 
# relation by running depth first search
function n_components(V, D)
    count = 0
    visited = Dict()
    for v in V
        visited[v] = false
    end
    function depth_first_search(k)
        visited[k] = true
        for p in V
            if !visited[p] && ((haskey(D, (k, p)) && D[(k, p)]) || (haskey(D, (p, k)) && D[(p, k)]))
                depth_first_search(p)
            end
        end
    end
    for p in V
        if !visited[p]
            depth_first_search(p)
            count += 1
        end
    end
    return count
end

@doc raw"""
    components(V::Vector{T}, D::Dict{Tuple{T, T}, Bool})
    
Outputs an array representing the connected components of the graph given by the vertices V and the edges D
(more precisely, the edges are given by the keys of D whose entries are "true").

# Example
julia> V = [1, 2, 3, 4];

julia> D = Dict{Tuple{Int, Int}, Bool}((1, 2) => true, (3, 4) => true, (2, 3) => false);

julia> components(V, D)
2-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
```
"""
function components(V, D)
    count = 0
    visited = Dict()
    for v in V
        visited[v] = false
    end
    # depth first search. We pass a component array as a parameter to store the components of the graph as we go along.
    function depth_first_search(k, component_arr)
        visited[k] = true
        push!(component_arr, k)
        for p in V
            # We need to check that the edge exists. D might be missing some keys so we need to check for that first.
            if !visited[p] && ((haskey(D, (k, p)) && D[(k, p)]) || (haskey(D, (p, k)) && D[(p, k)]))
                depth_first_search(p, component_arr)
            end
        end
    end
    components = []
    for p in V
        if !visited[p]
            component_arr = []
            depth_first_search(p, component_arr)
            push!(components, component_arr)
            count += 1
        end
    end
    return components
end

@doc raw"""
    enum_linear_regions_rat(f::TropicalPuiseuxPoly, g::TropicalPuiseuxPoly, verbose)

Computes the linear regions of a tropical Puiseux rational function f/g
Inputs: Tropical Puiseux polynomials f and g
Ouput: array containing linear regions of f/g represented by polyhedra/arrays of polyhedra.

# Example
Enumerates the linear regions of f/g where f = max(x, y) and g = max(x+y, x+2y).
```jldoctest
julia> f = TropicalPuiseuxPoly(Dict([1, 0] => 0, [0, 1] => 0), [[1, 0], [0, 1]]);

julia> g = TropicalPuiseuxPoly(Dict([1, 1] => 0, [1, 2] => 0), [[1, 1], [1, 2]]);

julia> enum_linear_regions_rat(f, g)
4-element Vector{Any}:
 Polyhedron in ambient dimension 2 with Float64 type coefficients
 Polyhedron in ambient dimension 2 with Float64 type coefficients
 Polyhedron in ambient dimension 2 with Float64 type coefficients
 Polyhedron in ambient dimension 2 with Float64 type coefficients
```
"""
function enum_linear_regions_rat(q::TropicalPuiseuxRational)
    f = q.num
    g = q.den
    # first, compute the linear regions of f and g. 
    lin_f = enum_linear_regions(f)
    lin_g = enum_linear_regions(g)
    # next, check which for repetitions of the linear map corresponding to f/g on intersections of the linear regions computed above.
    function check_linear_repetitions()
        linear_map = Dict()
        # We need to check for pairwise intersection of each polytope, by iterating over
        for i in eachindex(f)
            for j in eachindex(g)
                # we only need to do the checks on linear regions that are attained by f and g
                if lin_f[i][2] && lin_g[j][2]
                    # check if the polytopes intersect
                    poly = Oscar.intersect(lin_f[i][1], lin_g[j][1])
                    # if they intersect on a large enough region then add this to the list of linear maps that arise in f/g
                    # Note: we used to check that the poly is feasible and has dimension n, but it's a lot fast to check that it is full dimensional directly.
                    if Oscar.is_fulldimensional(poly)
                        linear_map[poly] = [Rational(f.coeff[f.exp[i]]) - Rational(g.coeff[g.exp[j]]), f.exp[i] - g.exp[j]]
                    end
                end
            end
        end
        # check for repetitions
        linear_map_unique = unique([l for (key, l) in linear_map])
        if length(linear_map) == length(linear_map_unique)
            return linear_map, [], false
        else
            # compute indices of repetitions for each linear map
            reps = [(l, Base.findall(x -> x == l, linear_map)) for l in linear_map_unique]
            return linear_map, reps, true
        end
    end
    linear_map, reps, exists_reps = check_linear_repetitions()
    # if there are no repetitions, then the linear regions are just the non-empty intersections of linear regions of f and linear regions of g
    if !exists_reps
        lin_regions = collect(keys(linear_map))
        # if there are repetitions then we will need to find connected components of the union of the polytopes on which repetitions occur.
    else
        # Initialise the array lin_regions. This will contain the true linear regions of f/g
        lin_regions = []
        # first find all pairwise intersections of polytopes.
        for (_, vals) in reps
            # if vals has length 1 then there is no other linear region with the same linear map
            if length(vals) == 1
                append!(lin_regions, vals)
            else
                # otherwise, we check for intersections in the set of linear regions with a given map
                has_intersect = Dict()
                # iterate over unordered pairs of (distinct) elements of vals
                for (poly1, poly2) in Combinatorics.combinations(vals, 2)
                    # intersect the two polyhedra
                    intesection = Oscar.intersect(poly1, poly2)
                    # add true to the dictionary if the intersection is nonemtpy and false otherwise
                    has_intersect[(poly1, poly2)] = Oscar.is_feasible(intesection)
                end
                # now find transitive closure of the relation given by dictionary has_intersect
                # and append the corresponding components to lin_regions
                append!(lin_regions, components(vals, has_intersect))
            end
        end
    end
    return lin_regions
end