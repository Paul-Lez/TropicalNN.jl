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
    LinearRegion{T}

Represents one linear region of a tropical Puiseux rational function.

A linear region is the maximal set on which the rational function restricts to a single
affine linear map. This set may be non-convex; `regions` holds all the full-dimensional
convex polyhedra whose union makes up the linear region.

Supports `length`, `iterate`, and integer indexing over `regions`.
"""
struct LinearRegion{T}
    regions::Vector{T}
end

Base.length(lr::LinearRegion) = length(lr.regions)
Base.iterate(lr::LinearRegion) = iterate(lr.regions)
Base.iterate(lr::LinearRegion, state) = iterate(lr.regions, state)
Base.getindex(lr::LinearRegion, i::Int) = lr.regions[i]

@doc raw"""
    LinearRegions{T}

The return type of `enum_linear_regions_rat`. Holds all linear regions of a tropical
Puiseux rational function as a vector of `LinearRegion{T}` objects.

Each element of `regions` is a `LinearRegion` corresponding to a distinct affine linear
map realised by the rational function. A `LinearRegion` may contain more than one convex
polyhedron when the same linear map is realised on several disconnected pieces.

Supports `length`, `iterate`, and integer indexing over the `LinearRegion` entries.
"""
struct LinearRegions{T}
    regions::Vector{LinearRegion{T}}
end

Base.length(lrs::LinearRegions) = length(lrs.regions)
Base.iterate(lrs::LinearRegions) = iterate(lrs.regions)
Base.iterate(lrs::LinearRegions, state) = iterate(lrs.regions, state)
Base.getindex(lrs::LinearRegions, i::Int) = lrs.regions[i]

@doc raw"""
    enum_linear_regions_rat(q::TropicalPuiseuxRational)

Computes the linear regions of a tropical Puiseux rational function.

# Arguments
- `q::TropicalPuiseuxRational`: The rational function whose linear regions are computed.

# Returns
A `LinearRegions` object whose `regions` field is a `Vector{LinearRegion}`. Each
`LinearRegion` corresponds to one distinct affine linear map realised by `q`, and its
`regions` field holds the full-dimensional convex polyhedra on which that map is attained.
When the same linear map appears on several disconnected pieces, all pieces are collected
in one `LinearRegion`.

# Example
Enumerates the linear regions of `f/g` where `f = max(x, y)` and `g = 0`.
`f/g` has two linear regions (one per monomial of `f`), each a single half-plane.
```jldoctest
julia> R = tropical_semiring(max);

julia> f = TropicalPuiseuxPoly([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]], false);

julia> g = TropicalPuiseuxPoly([R(0)], [[0//1, 0//1]], false);

julia> lr = enum_linear_regions_rat(f / g);

julia> length(lr)
2

julia> length(lr[1].regions)
1
```
"""
function enum_linear_regions_rat(q::TropicalPuiseuxRational)
    f = q.num
    g = q.den
    # first, compute the linear regions of f and g.
    lin_f = enum_linear_regions(f)
    lin_g = enum_linear_regions(g)
    # next, group full-dimensional intersections by the linear map they realise.
    function group_by_linear_map()
        groups = Dict()  # linear_map_value => Vector of polyhedra
        for i in eachindex(f)
            for j in eachindex(g)
                # only process linear regions that are attained by f and g
                if lin_f[i][2] && lin_g[j][2]
                    poly = Oscar.intersect(lin_f[i][1], lin_g[j][1])
                    if Oscar.is_fulldimensional(poly)
                        lm = [Rational(f.coeff[f.exp[i]]) - Rational(g.coeff[g.exp[j]]), f.exp[i] - g.exp[j]]
                        if haskey(groups, lm)
                            push!(groups[lm], poly)
                        else
                            groups[lm] = [poly]
                        end
                    end
                end
            end
        end
        return groups
    end

    groups = group_by_linear_map()
    lin_regions = [LinearRegion(polys) for (_, polys) in groups]
    return LinearRegions(lin_regions)
end