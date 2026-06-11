# This file contains code to compute 
# the linear regions of tropical Puiseux polynomials and rational functions.

# TODO: perhaps we want to allow this to vary?
const OSCAR_POLYHEDRON_COEFF_TYPE = Rational{BigInt}

function _constraint_scalar(::Type{Float64}, x)
    return Float64(Rational(x))
end

function _constraint_scalar(::Type{OSCAR_POLYHEDRON_COEFF_TYPE}, x)
    return OSCAR_POLYHEDRON_COEFF_TYPE(Rational(x))
end

function _constraint_vector(::Type{T}, values) where {T}
    return T[_constraint_scalar(T, value) for value in values]
end

function _constraint_matrix(::Type{T}, values) where {T}
    return Matrix{T}(map(value -> _constraint_scalar(T, value), values))
end

function _linear_region_constraints(
        f::AbstractSignomial,
        i,
        ::Type{T};
        include_self::Bool = false,
        empty_rows::Int = 1
) where {T}
    indices = [j for j in Base.eachindex(f) if include_self || j != i]
    if isempty(indices)
        return zeros(T, empty_rows, nvars(f)), zeros(T, empty_rows)
    end

    exp_i = _constraint_vector(T, get_exp(f, i))
    coeff_i = _constraint_scalar(T, get_coeff(f, i))
    rows = [_constraint_vector(T, get_exp(f, j)) - exp_i for j in indices]
    A = Matrix{T}(mapreduce(permutedims, vcat, rows))
    b = T[coeff_i - _constraint_scalar(T, get_coeff(f, j)) for j in indices]
    return A, b
end

@doc raw"""
    polyhedron(f::Signomial, i::Int)

Outputs the polyhedron corresponding to points where f is given by the
linear map corresponding to the i-th monomial of f.
"""
function polyhedron(f::AbstractSignomial, i)
    A, b = _linear_region_constraints(f, i, OSCAR_POLYHEDRON_COEFF_TYPE)
    return Oscar.polyhedron(A, b)
end

@doc raw"""
    enum_linear_regions(f::Signomial)

Outputs an array of tuples (poly, bool) indexed by the same set as the exponents of f. The tuple element poly
is the linear region corresponding to the exponent, and bool is true when this region is nonemtpy.

"""
function enum_linear_regions(f::AbstractSignomial)
    return map(Base.eachindex(f)) do i
        region = polyhedron(f, i)
        # add the polyhedron to the list plus a bool saying whether the polyhedron is non-empty
        # TODO: this should be replaced by a check that the polyhedron is full dimensional for performance.
        return (region, Oscar.is_feasible(region))
    end
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
            if !visited[p] &&
               ((haskey(D, (k, p)) && D[(k, p)]) || (haskey(D, (p, k)) && D[(p, k)]))
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
```jldoctest
julia> V = [1, 2, 3, 4];

julia> D = Dict{Tuple{Int, Int}, Bool}((1, 2) => true, (3, 4) => true, (2, 3) => false);

julia> components(V, D)
2-element Vector{Vector{Int64}}:
 [1, 2]
 [3, 4]
```
"""
function components(V, D)
    # Precompute adjacency list so DFS visits only actual neighbours — O(V+E)
    adj = Dict(v => eltype(V)[] for v in V)
    for ((u, v), connected) in D
        if connected
            haskey(adj, u) && push!(adj[u], v)
            haskey(adj, v) && push!(adj[v], u)
        end
    end

    visited = Dict(v => false for v in V)

    function depth_first_search(k, component_arr)
        visited[k] = true
        push!(component_arr, k)
        for p in adj[k]
            if !visited[p]
                depth_first_search(p, component_arr)
            end
        end
    end

    result = Vector{Vector{eltype(V)}}()
    for p in V
        if !visited[p]
            component_arr = eltype(V)[]
            depth_first_search(p, component_arr)
            push!(result, component_arr)
        end
    end
    return result
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
    enum_linear_regions_rat(q::RationalSignomial)

Computes the linear regions of a tropical Puiseux rational function.

# Arguments
- `q::RationalSignomial`: The rational function whose linear regions are computed.

# Returns
A `LinearRegions` object whose `regions` field is a `Vector{LinearRegion}`. Each
`LinearRegion` corresponds to one distinct affine linear map realised by `q`, and its
`regions` field holds the full-dimensional convex polyhedra on which that map is attained.
When the same linear map appears on several disconnected pieces, all pieces are collected
in one `LinearRegion`.

"""
function enum_linear_regions_rat(q::RationalSignomial)
    f = q.num
    g = q.den
    # first, compute the linear regions of f and g.
    lin_f = enum_linear_regions(f)
    lin_g = enum_linear_regions(g)
    # next, group full-dimensional intersections by the linear map they realise.
    function group_by_linear_map()
        groups = Dict()  # linear_map_value => Vector of polyhedra
        for i in Base.eachindex(f)
            for j in Base.eachindex(g)
                # only process linear regions that are attained by f and g
                if lin_f[i][2] && lin_g[j][2]
                    poly = Oscar.intersect(lin_f[i][1], lin_g[j][1])
                    if Oscar.is_fulldimensional(poly)
                        lm = (Rational(get_coeff(f, i)) - Rational(get_coeff(g, j)),
                            collect(get_exp(f, i)) - collect(get_exp(g, j)))
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
