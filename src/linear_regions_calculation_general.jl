# Shared implementation for computing linear regions of tropical Puiseux
# polynomials and rational functions.

const OSCAR_POLYHEDRON_COEFF_TYPE = Rational{BigInt}
const HIGHS_DEFAULT_TOL = 1e-6

"""
    LinearRegionsCalculationMode

Abstract supertype for backend selectors used by the mode-based linear-region
API.
"""
abstract type LinearRegionsCalculationMode end

struct _Oscar <: LinearRegionsCalculationMode end

Base.@kwdef struct _HiGHS <: LinearRegionsCalculationMode
    tol::Float64 = HIGHS_DEFAULT_TOL
    solver::String = "hipo"
end

"""
    OscarMode()

Use Oscar polyhedra and exact rational arithmetic for linear-region calculations.
"""
const OscarMode = _Oscar

"""
    HiGHSMode(; tol=HIGHS_DEFAULT_TOL, solver="hipo")

Use JuMP/HiGHS LP checks and floating-point constraint matrices for
linear-region calculations. `tol` controls full-dimensionality checks.
"""
const HiGHSMode = _HiGHS

struct _Polyhedra
    A::Matrix{Float64}
    b::Vector{Float64}
end

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

The return type of `enum_linear_regions_rat_general`. Holds all linear regions
of a tropical Puiseux rational function as a vector of `LinearRegion{T}` objects.

Each element of `regions` is a `LinearRegion` corresponding to a distinct affine linear
map realised by the rational function on one connected component. A
`LinearRegion` may contain more than one convex polyhedron when adjacent
full-dimensional pieces realise the same map.

Supports `length`, `iterate`, and integer indexing over the `LinearRegion` entries.
"""
struct LinearRegions{T}
    regions::Vector{LinearRegion{T}}
end

Base.length(lrs::LinearRegions) = length(lrs.regions)
Base.iterate(lrs::LinearRegions) = iterate(lrs.regions)
Base.iterate(lrs::LinearRegions, state) = iterate(lrs.regions, state)
Base.getindex(lrs::LinearRegions, i::Int) = lrs.regions[i]

# Computes the number of equivalence classes of the transitive closure of a
# relation by running depth first search.
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
    adj = Dict(v => eltype(V)[] for v in V)
    for ((u, v), connected) in D
        if connected
            # Ignore edges whose endpoints are not both in the vertex set V.
            if haskey(adj, u) && haskey(adj, v)
                push!(adj[u], v)
                push!(adj[v], u)
            end
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

function create_highs_model(; solver = "hipo")
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "solver", solver)
    return model
end

"""
    highs_is_empty(A::Matrix{Float64}, b::Vector{Float64}; solver="hipo")

Check if polyhedron `{x : Ax <= b}` is empty via LP feasibility using HiGHS.
"""
function highs_is_empty(A::Matrix{Float64}, b::Vector{Float64}; solver = "hipo")
    _, n = size(A)

    model = create_highs_model(; solver = solver)

    @variable(model, x[1:n])
    @constraint(model, A * x .<= b)

    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        return false
    elseif status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
        return true
    end
    throw(ErrorException("HiGHS feasibility check ended with unexpected status $status"))
end

"""
    highs_is_full_dimensional(A::Matrix{Float64}, b::Vector{Float64}; tol=HIGHS_DEFAULT_TOL, solver="hipo")

Check if polyhedron `{x : Ax <= b}` is full dimensional via LP using HiGHS.
"""
function highs_is_full_dimensional(
        A::Matrix{Float64},
        b::Vector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = "hipo"
)
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))

    m, n = size(A)

    zero_rows = [i for i in 1:m if all(iszero, A[i, :])]
    any(i -> b[i] < 0, zero_rows) && return false
    non_trivial_rows = [i for i in 1:m if !(i in zero_rows)]

    if isempty(non_trivial_rows)
        return true
    end

    A_filtered = A[non_trivial_rows, :]
    b_filtered = b[non_trivial_rows]

    model = create_highs_model(; solver = solver)

    @variable(model, x[1:n])
    @variable(model, ε)
    @constraint(model, A_filtered * x .+ ε .<= b_filtered)
    @objective(model, Max, ε)

    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        epsilon_value = value(ε)
        isfinite(epsilon_value) ||
            throw(ErrorException("HiGHS returned non-finite inflation value $epsilon_value"))
        return epsilon_value > tol
    elseif status == MOI.DUAL_INFEASIBLE
        return true
    elseif status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
        return false
    end
    throw(ErrorException("HiGHS full-dimensionality check ended with unexpected status $status"))
end

"""
    highs_intersect_is_full_dimensional(A1::Matrix{Float64}, b1::Vector{Float64},
                                        A2::Matrix{Float64}, b2::Vector{Float64};
                                        tol=HIGHS_DEFAULT_TOL, solver="hipo")

Check if the intersection of two polyhedra is full dimensional via LP using HiGHS.
"""
function highs_intersect_is_full_dimensional(A1::Matrix{Float64}, b1::Vector{Float64},
        A2::Matrix{Float64}, b2::Vector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = "hipo")
    size(A1, 2) == size(A2, 2) ||
        throw(DimensionMismatch("Ambient dimensions must match, got $(size(A1, 2)) and $(size(A2, 2))"))

    A_combined = vcat(A1, A2)
    b_combined = vcat(b1, b2)
    return highs_is_full_dimensional(A_combined, b_combined; tol = tol, solver = solver)
end

function make_polyhedron(A, b; mode::LinearRegionsCalculationMode)
    if mode isa _Oscar
        A_exact = _constraint_matrix(OSCAR_POLYHEDRON_COEFF_TYPE, A)
        b_exact = _constraint_vector(OSCAR_POLYHEDRON_COEFF_TYPE, b)
        return Oscar.polyhedron(A_exact, b_exact)
    elseif mode isa _HiGHS
        A_float = _constraint_matrix(Float64, A)
        b_float = _constraint_vector(Float64, b)
        return _Polyhedra(A_float, b_float)
    end
    throw(ArgumentError("Unsupported linear-regions calculation mode $(typeof(mode))"))
end

"""
    get_matrix(region; mode)

Return the halfspace matrix `A` for a backend region representation of
`{x : Ax <= b}`.
"""
function get_matrix(region::Oscar.Polyhedron; mode::_Oscar)
    return Float64.(Oscar.halfspace_matrix_pair(Oscar.facets(region)).A)
end

function get_matrix(region::_Polyhedra; mode::_HiGHS)
    return region.A
end

"""
    get_vector(region; mode)

Return the halfspace vector `b` for a backend region representation of
`{x : Ax <= b}`.
"""
function get_vector(region::Oscar.Polyhedron; mode::_Oscar)
    return Float64.(Oscar.halfspace_matrix_pair(Oscar.facets(region)).b)
end

function get_vector(region::_Polyhedra; mode::_HiGHS)
    return region.b
end

function is_feasible(region::Oscar.Polyhedron; mode::_Oscar)
    return Oscar.is_feasible(region)
end

function is_feasible(region::_Polyhedra; mode::_HiGHS)
    return !highs_is_empty(region.A, region.b; solver = mode.solver)
end

function is_full_dimensional(region::Oscar.Polyhedron; mode::_Oscar)
    return Oscar.is_fulldimensional(region)
end

function is_full_dimensional(region::_Polyhedra; mode::_HiGHS)
    return highs_is_full_dimensional(
        region.A,
        region.b;
        tol = mode.tol,
        solver = mode.solver
    )
end

@doc raw"""
    polyhedron(f::AbstractSignomial, i::Int, mode::LinearRegionsCalculationMode)

Outputs the polyhedron corresponding to points where f is given by the
linear map corresponding to the i-th monomial of f, using the selected backend
mode.
"""
function polyhedron(f::AbstractSignomial, i, mode::LinearRegionsCalculationMode)
    if mode isa _Oscar
        coefficient_type = OSCAR_POLYHEDRON_COEFF_TYPE
    else
        coefficient_type = Float64
    end

    if mode isa _HiGHS
        empty_rows = 0
    else
        empty_rows = 1
    end

    A, b = _linear_region_constraints(f, i, coefficient_type; empty_rows = empty_rows)
    # The polyhedron is then the set of points x such that Ax ≤ b.
    return make_polyhedron(A, b; mode = mode)
end

"""
    region_intersection(region_1, region_2; mode)

Return the `mode` representation of `region_1 ∩ region_2`.
"""
function region_intersection(region_1, region_2; mode::LinearRegionsCalculationMode)
    A = vcat(get_matrix(region_1; mode = mode), get_matrix(region_2; mode = mode))
    b = vcat(get_vector(region_1; mode = mode), get_vector(region_2; mode = mode))
    return make_polyhedron(A, b; mode = mode)
end

function region_intersection(
        region_1::Oscar.Polyhedron,
        region_2::Oscar.Polyhedron;
        mode::_Oscar
)
    return Oscar.intersect(region_1, region_2)
end

"""
    regions_intersect(region_1, region_2; mode) -> Bool

Return whether two regions have nonempty intersection.
"""
function regions_intersect(region_1, region_2; mode::LinearRegionsCalculationMode)
    return is_feasible(region_intersection(region_1, region_2; mode = mode); mode = mode)
end

"""
    enum_linear_regions_general(f::AbstractSignomial; mode)

Compute all linear-region candidates for signomial `f` using the selected
backend mode.

Returns a vector of `(region, is_feasible)` tuples indexed by the monomials of
`f`.
"""
function enum_linear_regions_general(
        f::AbstractSignomial;
        mode::LinearRegionsCalculationMode
)
    return map(Base.eachindex(f)) do i
        region = polyhedron(f, i, mode)
        return (region, is_feasible(region; mode = mode))
    end
end

function _linear_map_key(f::AbstractSignomial, g::AbstractSignomial, i, j)
    coeff = Rational(get_coeff(f, i)) - Rational(get_coeff(g, j))
    exp = collect(get_exp(f, i)) - collect(get_exp(g, j))
    return (coeff, exp)
end

"""
    enum_linear_regions_rat_general(q::RationalSignomial; mode)

Compute the linear regions of a tropical Puiseux rational function using the
algorithm indicated by `mode`.
"""
function enum_linear_regions_rat_general(
        q::RationalSignomial;
        mode::LinearRegionsCalculationMode
)
    f = q.num
    g = q.den
    length(f) > 0 ||
        throw(ArgumentError("RationalSignomial numerator must have at least one monomial"))
    length(g) > 0 ||
        throw(ArgumentError("RationalSignomial denominator must have at least one monomial"))

    lin_f = enum_linear_regions_general(f; mode = mode)
    lin_g = enum_linear_regions_general(g; mode = mode)
    region_type = typeof(lin_f[begin][1])
    if region_type != typeof(lin_g[begin][1])
        throw(ArgumentError(
            "Numerator and denominator regions use incompatible representations: " *
            "$region_type and $(typeof(lin_g[begin][1]))",
        ))
    end
    map_to_regions = Dict{Any, Vector{region_type}}()

    for i in Base.eachindex(f)
        for j in Base.eachindex(g)
            if lin_f[i][2] && lin_g[j][2]
                intersection = region_intersection(lin_f[i][1], lin_g[j][1]; mode = mode)
                if is_full_dimensional(intersection; mode = mode)
                    key = _linear_map_key(f, g, i, j)
                    if haskey(map_to_regions, key)
                        push!(map_to_regions[key], intersection)
                    else
                        map_to_regions[key] = region_type[intersection]
                    end
                end
            end
        end
    end

    linear_regions = LinearRegion{region_type}[]
    for regions in values(map_to_regions)
        region_components = if length(regions) == 1
            (regions,)
        else
            has_intersection = Dict()
            for (region_1, region_2) in Combinatorics.combinations(regions, 2)
                has_intersection[(region_1, region_2)] = regions_intersect(region_1, region_2; mode = mode)
            end
            # TODO: eventually we should replace `components` by one of the functions here
            # https://juliagraphs.org/Graphs.jl/stable/algorithms/connectivity/
            components(regions, has_intersection)
        end

        for component in region_components
            push!(linear_regions, LinearRegion(convert(Vector{region_type}, component)))
        end
    end

    isempty(linear_regions) &&
        throw(ArgumentError("No full-dimensional linear regions were found for the rational signomial"))
    return LinearRegions(linear_regions)
end
