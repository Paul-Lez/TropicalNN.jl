# Shared implementation for computing linear regions of tropical Puiseux
# polynomials and rational functions.

const OSCAR_POLYHEDRON_COEFF_TYPE = Rational{BigInt}
const HIGHS_DEFAULT_TOL = 1e-6
const HIGHS_DEFAULT_SOLVER = "choose"

"""
    _assert_tropicalnn_loaded(pool)

Check that the TropicalNN module is loaded on all workers in `pool`.
"""
function _assert_tropicalnn_loaded(pool::Distributed.AbstractWorkerPool)
    unloaded = filter(Distributed.workers(pool)) do pid
        !Distributed.remotecall_fetch(isdefined, pid, Main, :TropicalNN)
    end

    isempty(unloaded) || throw(ArgumentError(
        "TropicalNN is not loaded on worker(s): $(join(unloaded, ", ")). " *
        "Run `@everywhere using TropicalNN` after adding workers.",
    ))
end

"""
    _index_chunks(n, nworkers)

Partition `1:n` into contiguous chunks sized for `nworkers` workers.
The result contains at most four chunks per worker.
"""
function _index_chunks(n::Int, nworkers::Int)
    n <= 0 && return UnitRange{Int}[]

    nchunks = min(n, 4 * max(1, nworkers))
    chunk_size = cld(n, nchunks)
    return [start:min(start + chunk_size - 1, n) for start in 1:chunk_size:n]
end

"""
    LinearRegionsCalculationMode

Backend selector for linear-region calculations.
"""
abstract type LinearRegionsCalculationMode end

"""
    _Oscar

Use Oscar's exact polyhedra and rational arithmetic for linear-region calculations.
"""
struct _Oscar <: LinearRegionsCalculationMode end

"""
    _HiGHS(; tol=HIGHS_DEFAULT_TOL, solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Use HiGHS to check feasibility and full-dimensionality of linear regions.
`tol` sets the full-dimensionality tolerance. `threads` sets the optional
HiGHS thread count.
"""
Base.@kwdef struct _HiGHS <: LinearRegionsCalculationMode
    tol::Float64 = HIGHS_DEFAULT_TOL
    solver::String = HIGHS_DEFAULT_SOLVER
    threads::Union{Nothing, Int} = nothing
end

"""
    OscarMode()

Use Oscar polyhedra and exact rational constraints.
"""
const OscarMode = _Oscar

"""
    HiGHSMode(; tol=HIGHS_DEFAULT_TOL, solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Use JuMP and HiGHS with `Float64` constraints. `tol` sets the
full-dimensionality tolerance, and `threads` sets the optional solver thread
count.
"""
const HiGHSMode = _HiGHS

"""
    _Polyhedra

Store a HiGHS-mode polyhedron in halfspace form `{x : Ax ≤ b}`.
"""
struct _Polyhedra
    A::Matrix{Float64}
    b::Vector{Float64}
end

"""
    _constraint_scalar(T, x)

Convert a coefficient or exponent entry `x` to the scalar type `T` used by a
linear-region backend.
"""
function _constraint_scalar(::Type{Float64}, x::Oscar.TropicalSemiringElem)
    return Float64(Rational(x))
end

function _constraint_scalar(::Type{Float64}, x::Real)
    return Float64(x)
end

function _constraint_scalar(::Type{Float64}, x)
    return Float64(Rational(x))
end

function _constraint_scalar(::Type{OSCAR_POLYHEDRON_COEFF_TYPE}, x::AbstractFloat)
    return OSCAR_POLYHEDRON_COEFF_TYPE(x)
end

function _constraint_scalar(::Type{OSCAR_POLYHEDRON_COEFF_TYPE}, x)
    return OSCAR_POLYHEDRON_COEFF_TYPE(Rational(x))
end

"""
    _constraint_vector(T, values)

Convert `values` to a vector with element type `T` for a constraint system.
"""
function _constraint_vector(::Type{T}, values) where {T}
    return T[_constraint_scalar(T, value) for value in values]
end

"""
    _constraint_matrix(T, values)

Convert `values` to a matrix with element type `T` for a constraint system.
"""
function _constraint_matrix(::Type{T}, values) where {T}
    return Matrix{T}(map(value -> _constraint_scalar(T, value), values))
end

"""
    _linear_region_constraints(f, i, T; include_self=false,
                               competitors=eachindex(f))

Construct the halfspace system `{x : Ax ≤ b}` on which the `i`th monomial of
`f` dominates the selected competing monomials. `T` determines the constraint
coefficient type. If there are no competitors, return the zero-row system that
represents the full ambient space.
"""
function _linear_region_constraints(
        f::Signomial,
        i,
        ::Type{T};
        include_self::Bool = false,
        competitors = Base.eachindex(f)
) where {T}
    indices = [j for j in competitors if include_self || j != i]
    if isempty(indices)
        return zeros(T, 0, nvars(f)), T[]
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

One linear region of a tropical signomial or rational signomial.
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

Result of `linear_regions`. Each element is a `LinearRegion`.
"""
struct LinearRegions{T}
    regions::Vector{LinearRegion{T}}
end

Base.length(lrs::LinearRegions) = length(lrs.regions)
Base.iterate(lrs::LinearRegions) = iterate(lrs.regions)
Base.iterate(lrs::LinearRegions, state) = iterate(lrs.regions, state)
Base.getindex(lrs::LinearRegions, i::Int) = lrs.regions[i]

"""
    _components_graph(V, D)

Construct the undirected graph with vertices `V` and the true-valued edges in
the adjacency dictionary `D`.
"""
function _components_graph(V, D)
    vertex_to_index = Dict(v => i for (i, v) in pairs(V))
    graph = Graphs.SimpleGraph(length(V))

    for ((u, v), connected) in D
        if connected
            u_idx = vertex_to_index[u]
            v_idx = vertex_to_index[v]
            if u_idx != v_idx
                Graphs.add_edge!(graph, u_idx, v_idx)
            end
        end
    end

    return graph
end

@doc raw"""
    components(V::Vector{T}, D::Dict{Tuple{T, T}, Bool})

Return the connected components of the graph with vertices `V` and edges in `D`.

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
    graph = _components_graph(V, D)
    return [V[component] for component in Graphs.connected_components(graph)]
end

"""
    create_highs_model(; solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Create a HiGHS model configured with the requested solver and optional
thread count.
"""
function create_highs_model(; solver = HIGHS_DEFAULT_SOLVER, threads = nothing)
    if threads !== nothing && threads < 1
        throw(ArgumentError("threads must be at least 1, got $threads"))
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    set_attribute(model, "solver", solver)
    if threads !== nothing
        HiGHS.Highs_resetGlobalScheduler(1)
        set_attribute(model, MOI.NumberOfThreads(), threads)
    end
    return model
end

"""
    highs_is_empty(A::Matrix{Float64}, b::Vector{Float64}; solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Check if polyhedron `{x : Ax <= b}` is empty by solving a linear program using HiGHS.
"""
function highs_is_empty(
        A::Matrix{Float64},
        b::Vector{Float64};
        solver = HIGHS_DEFAULT_SOLVER,
        threads = nothing
)
    _, n = size(A)

    model = create_highs_model(; solver = solver, threads = threads)

    @variable(model, x[1:n])
    @constraint(model, A * x .<= b)
    # Stop when JuMP finds a feasible point.
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
    filter_lp(A, b)

Remove satisfied constant inequalities from `{x : Ax ≤ b}`. Return `nothing`
when a constant inequality is infeasible.
"""
function filter_lp(A::Matrix{T}, b::Vector{T}) where {T}
    m, n = size(A)

    zero_rows = [i for i in 1:m if all(iszero, A[i, :])]
    any(i -> b[i] < 0, zero_rows) && return nothing
    non_trivial_rows = [i for i in 1:m if !(i in zero_rows)]

    A_filtered = A[non_trivial_rows, :]
    b_filtered = b[non_trivial_rows]

    return A_filtered, b_filtered
end

"""
    highs_is_full_dimensional(A::Matrix{Float64}, b::Vector{Float64}; tol=HIGHS_DEFAULT_TOL, solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Check if polyhedron `{x : Ax <= b}` is full dimensional by solving a linear program using HiGHS.
"""
function highs_is_full_dimensional(
        A::Matrix{Float64},
        b::Vector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = HIGHS_DEFAULT_SOLVER,
        threads = nothing
)
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))

    m, n = size(A)

    filtered = filter_lp(A, b)
    # An infeasible polyhedron is not full-dimensional.
    if filtered === nothing
        return false
    end
    A_filtered, b_filtered = filtered

    # With no nonconstant inequalities, the polyhedron is all of R^n.
    if isempty(b_filtered)
        return true
    end

    model = create_highs_model(; solver = solver, threads = threads)

    @variable(model, x[1:n])
    @variable(model, ε)
    @constraints(model, begin
        A_filtered * x .+ ε .<= b_filtered
        # Bound the slack objective.
        ε <= 1
    end)
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
                                        tol=HIGHS_DEFAULT_TOL, solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Check if the intersection of two polyhedra is full dimensional via by solving linear programs using HiGHS.
"""
function highs_intersect_is_full_dimensional(A1::Matrix{Float64}, b1::Vector{Float64},
        A2::Matrix{Float64}, b2::Vector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = HIGHS_DEFAULT_SOLVER,
        threads = nothing)
    size(A1, 2) == size(A2, 2) ||
        throw(DimensionMismatch("Ambient dimensions must match, got $(size(A1, 2)) and $(size(A2, 2))"))

    A_combined = vcat(A1, A2)
    b_combined = vcat(b1, b2)
    return highs_is_full_dimensional(
        A_combined,
        b_combined;
        tol = tol,
        solver = solver,
        threads = threads
    )
end

"""
    highs_check_implicit_equality(A, b, i; tol, solver, threads)

Check whether the `i`th inequality of `{x : Ax ≤ b}` is an implicit equality,
using a HiGHS linear program.
"""
function highs_check_implicit_equality(
        A::Matrix{Float64}, b::Vector{Float64}, i::Int;
        tol = HIGHS_DEFAULT_TOL,
        solver = HIGHS_DEFAULT_SOLVER,
        threads = nothing
)
    m, n = size(A)
    1 <= i <= m || throw(BoundsError("Row index $i out of bounds for matrix with $m rows"))

    model = create_highs_model(; solver = solver, threads = threads)

    @variable(model, x[1:n])
    @variable(model, s)
    @constraints(model, begin
        A * x .<= b
        s == b[i] - LinearAlgebra.dot(A[i, :], x)
    end)
    # Maximize the slack of row i. Zero maximum slack means an implicit equality.
    @objective(model, Max, s)

    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        s_value = value(s)
        return !(s_value > tol)
    elseif status == MOI.INFEASIBLE_OR_UNBOUNDED || status == MOI.DUAL_INFEASIBLE
        # An unbounded slack means that row i is not an implicit equality.
        return false
    else
        # The constraint system was feasible before this check.
        throw(ErrorException("HiGHS full-dimensionality check ended with unexpected status $status"))
    end
end

"""
    _highs_codimension_le_one(A, b; tol, solver, threads)

Return whether the HiGHS-mode polyhedron `{x : Ax ≤ b}` has codimension at
most one.
"""
function _highs_codimension_le_one(A::Matrix{Float64},
        b::Vector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = HIGHS_DEFAULT_SOLVER,
        threads = nothing
)

    # Remove constant inequalities from the LP.
    filtered = filter_lp(A, b)
    # An infeasible polyhedron does not have codimension at most one.
    if filtered === nothing
        return false
    end
    A_filtered, b_filtered = filtered

    if highs_is_empty(A_filtered, b_filtered; solver = solver, threads = threads)
        return false
    end

    if highs_is_full_dimensional(
        A_filtered, b_filtered; tol = tol, solver = solver, threads = threads)
        return true
    end

    m, n = size(A)

    redundantIdx = []

    for i in 1:size(A_filtered, 1)
        # Two independent implicit equalities give codimension at least two.
        if highs_check_implicit_equality(
            A_filtered, b_filtered, i; tol = tol, solver = solver, threads = threads)
            push!(redundantIdx, i)
        end
        if length(redundantIdx) > 1
            # Test independence with the row rank.
            A_redundant = A_filtered[redundantIdx, :]
            if rank(A_redundant) > 1
                return false
            end
        end
    end
    return true
end

"""
    codimension_le_one(A, b; mode)

Return whether `{x : Ax ≤ b}` has codimension at most one.
"""
function codimension_le_one(A, b; mode::LinearRegionsCalculationMode)
    if mode isa _Oscar
        poly = make_polyhedron(A, b; mode = mode)
        return Oscar.codim(poly) <= 1
    elseif mode isa _HiGHS
        return _highs_codimension_le_one(
            A,
            b;
            tol = mode.tol,
            solver = mode.solver,
            threads = mode.threads
        )
    end
end

"""
    regions_intersect_codimension_le_one(region_1, region_2; mode)

Return whether two regions meet in codimension zero or one.
"""
function regions_intersect_codimension_le_one(region_1, region_2; mode::LinearRegionsCalculationMode)
    A = vcat(get_matrix(region_1; mode = mode), get_matrix(region_2; mode = mode))
    b = vcat(get_vector(region_1; mode = mode), get_vector(region_2; mode = mode))
    return codimension_le_one(A, b; mode = mode)
end

"""
    make_polyhedron(A, b; mode)

Construct the selected backend representation of the halfspace system
`{x : Ax ≤ b}`.
"""
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
    _linear_region_coefficient_type(mode)

Return the scalar type used to construct linear-region constraints for `mode`.
"""
function _linear_region_coefficient_type(mode::LinearRegionsCalculationMode)
    if mode isa _Oscar
        return OSCAR_POLYHEDRON_COEFF_TYPE
    else
        return Float64
    end
end

"""
    _linear_region_constraint_data(f, i, mode; competitors=eachindex(f))

Return the backend-typed constraint matrix and vector for the dominance region
of the `i`th monomial of `f`.
"""
function _linear_region_constraint_data(
        f::Signomial,
        i,
        mode::LinearRegionsCalculationMode;
        competitors = Base.eachindex(f)
)
    return _linear_region_constraints(
        f,
        i,
        _linear_region_coefficient_type(mode);
        competitors = competitors
    )
end

"""
    get_matrix(region; mode)

Return `A` from the halfspace form `{x : A * x ≤ b}`.
"""
function get_matrix(region::Oscar.Polyhedron; mode::_Oscar)
    return Float64.(Oscar.halfspace_matrix_pair(Oscar.facets(region)).A)
end

function get_matrix(region::_Polyhedra; mode::_HiGHS)
    return region.A
end

"""
    get_vector(region; mode)

Return `b` from the halfspace form `{x : A * x ≤ b}`.
"""
function get_vector(region::Oscar.Polyhedron; mode::_Oscar)
    return Float64.(Oscar.halfspace_matrix_pair(Oscar.facets(region)).b)
end

function get_vector(region::_Polyhedra; mode::_HiGHS)
    return region.b
end

"""
    is_feasible(region; mode)

Return whether `region` is nonempty in the selected backend.
"""
function is_feasible(region::Oscar.Polyhedron; mode::_Oscar)
    return Oscar.is_feasible(region)
end

function is_feasible(region::_Polyhedra; mode::_HiGHS)
    return !highs_is_empty(region.A, region.b; solver = mode.solver, threads = mode.threads)
end

"""
    is_full_dimensional(region; mode)

Return whether `region` has the ambient dimension in the selected backend.
"""
function is_full_dimensional(region::Oscar.Polyhedron; mode::_Oscar)
    return Oscar.is_fulldimensional(region)
end

function is_full_dimensional(region::_Polyhedra; mode::_HiGHS)
    return highs_is_full_dimensional(
        region.A,
        region.b;
        tol = mode.tol,
        solver = mode.solver,
        threads = mode.threads
    )
end

@doc raw"""
    polyhedron(f::Signomial, i::Int, mode::LinearRegionsCalculationMode; competitors=Base.eachindex(f))

Return the polyhedron where monomial `i` of `f` attains the maximum.
`competitors` selects the other monomials used in the comparisons.
"""
function polyhedron(
        f::Signomial,
        i,
        mode::LinearRegionsCalculationMode;
        competitors = Base.eachindex(f)
)
    A, b = _linear_region_constraint_data(f, i, mode; competitors = competitors)
    # The polyhedron is {x : A * x ≤ b}.
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
    _linear_region_data((f, i, mode))

Construct the constraint data and feasibility flag for the `i`th monomial of
`f`. The input is a tuple so we can directly pass this to `pmap`.
"""
function _linear_region_data(args)
    f, i, mode = args
    A, b = _linear_region_constraint_data(f, i, mode)
    region = make_polyhedron(A, b; mode = mode)
    return (A, b, is_feasible(region; mode = mode))
end

"""
    _linear_region_data_chunk((f, indices, mode))

Evaluate `_linear_region_data` for a contiguous collection of monomial
indices.
The input is a tuple so we can directly pass this to `pmap`.
"""
function _linear_region_data_chunk(args)
    f, inds, mode = args
    return [_linear_region_data((f, i, mode)) for i in inds]
end

"""
    _linear_region_data_parallel(f, mode, workers)

Return dominance-region constraint data for every monomial of `f`. When
`workers` is provided, process index chunks with `Distributed.pmap`.
"""
function _linear_region_data_parallel(
        f::Signomial,
        mode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool}
)
    n = length(f)
    if workers === nothing || n <= 1
        return [_linear_region_data((f, i, mode)) for i in Base.eachindex(f)]
    end

    _assert_tropicalnn_loaded(workers)
    chunks = _index_chunks(n, length(Distributed.workers(workers)))
    chunk_results = Distributed.pmap(
        _linear_region_data_chunk,
        workers,
        [(f, chunk, mode) for chunk in chunks]
    )
    return Base.reduce(vcat, chunk_results)
end

"""
    linear_regions(f::Signomial; mode, workers=nothing)

Return the nonempty dominance polyhedra of `f` as `(index, region)` pairs.
The results can include lower-dimensional regions.
"""
function linear_regions(
        f::Signomial;
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    vector_regions = linear_regions([f]; mode = mode, workers = workers)
    return [(only(index), region) for (index, region) in vector_regions]
end

"""
    linear_regions(f::AbstractVector{<:Signomial}; mode, workers=nothing)

Return the common nonempty partition of `f` as `(indices, region)` pairs.
"""
function linear_regions(
        f::AbstractVector{<:Signomial};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    regions = []
    isempty(f) && return regions

    linear_regions_vec = map(f) do signomial
        region_data = _linear_region_data_parallel(signomial, mode, workers)
        return [(i, make_polyhedron(data[1], data[2]; mode = mode))
                for (i, data) in pairs(region_data) if data[3]]
    end

    for candidates in Iterators.product(linear_regions_vec...)
        index = map(first, candidates)
        candidate_regions = map(last, candidates)
        region = Base.reduce(
            (region_1, region_2) -> region_intersection(region_1, region_2; mode = mode),
            candidate_regions
        )
        if length(candidate_regions) == 1 || is_feasible(region; mode = mode)
            push!(regions, (index, region))
        end
    end

    return regions
end

"""
    _linear_map_key(f, g, i, j)

Return a hashable representation of the affine map obtained by subtracting
the `j`th monomial of `g` from the `i`th monomial of `f`.
"""
function _linear_map_key(f::Signomial, g::Signomial, i, j)
    coeff = Rational(get_coeff(f, i)) - Rational(get_coeff(g, j))
    exp = collect(get_exp(f, i)) - collect(get_exp(g, j))
    return (coeff, exp)
end

function _linear_map_key(f::Vector{<:Signomial}, g::Vector{<:Signomial}, idxf, idxg)
    # Each vector has one entry for each output coordinate.
    @assert length(f) == length(g) == length(idxf) == length(idxg)
    return map(i -> _linear_map_key(f[i], g[i], idxf[i], idxg[i]), Base.eachindex(idxf))
end

"""
    _region_constraint_data(region; mode)

Return the halfspace matrix and vector for `region`, using rational
coefficients when the Oscar backend is used. Encode each affine-hull equation
as two inequalities.
"""
function _region_constraint_data(region::Oscar.Polyhedron; mode::_Oscar)
    facets = Oscar.halfspace_matrix_pair(Oscar.facets(region))
    affine_hull = Oscar.halfspace_matrix_pair(Oscar.affine_hull(region))
    return (
        vcat(facets.A, affine_hull.A, -affine_hull.A),
        vcat(facets.b, affine_hull.b, -affine_hull.b)
    )
end

function _region_constraint_data(region::_Polyhedra; mode::_HiGHS)
    return (region.A, region.b)
end

"""
    _partition_constraint_data(regions; mode)

Convert `(index, region)` partition entries to `(index, A, b)` entries for
intersection checks and distributed transport.
"""
function _partition_constraint_data(regions; mode::LinearRegionsCalculationMode)
    return map(regions) do entry
        index, region = entry
        A, b = _region_constraint_data(region; mode = mode)
        return (index, A, b)
    end
end

"""
    _rational_region_intersections_chunk((f, g, numerator, denominator, pairs, mode))

Test the requested numerator/denominator partition pairs for full-dimensional
intersection. Return the affine-map key and constraint data for each accepted
intersection.
The input is a tuple so we can directly pass this to `pmap`.
"""
function _rational_region_intersections_chunk(args)
    f, g, lin_f, lin_g, pairs, mode = args
    intersections = Vector{Tuple{Any, Any, Any}}()

    for (i, j) in pairs
        idx_f, A_f, b_f = lin_f[i]
        idx_g, A_g, b_g = lin_g[j]
        A = vcat(A_f, A_g)
        b = vcat(b_f, b_g)
        intersection = make_polyhedron(A, b; mode = mode)
        if is_full_dimensional(intersection; mode = mode)
            push!(intersections, (_linear_map_key(f, g, idx_f, idx_g), A, b))
        end
    end

    return intersections
end

"""
    _rational_region_intersections_parallel(f, g, numerator, denominator, mode, workers)

Test every numerator/denominator partition pair for full-dimensional
intersection. Use `workers` to evaluate pair chunks in parallel when supplied.
"""
function _rational_region_intersections_parallel(
        f,
        g,
        lin_f,
        lin_g,
        mode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool}
)
    pairs = Tuple{Int, Int}[]
    for i in Base.eachindex(lin_f)
        for j in Base.eachindex(lin_g)
            push!(pairs, (i, j))
        end
    end

    isempty(pairs) && return Tuple{Any, Any, Any}[]
    if workers === nothing || length(pairs) <= 1
        return _rational_region_intersections_chunk((f, g, lin_f, lin_g, pairs, mode))
    end

    _assert_tropicalnn_loaded(workers)
    chunks = _index_chunks(length(pairs), length(Distributed.workers(workers)))
    pair_chunks = [pairs[chunk] for chunk in chunks]
    chunk_results = Distributed.pmap(
        _rational_region_intersections_chunk,
        workers,
        [(f, g, lin_f, lin_g, pair_chunk, mode) for pair_chunk in pair_chunks]
    )
    return Base.reduce(vcat, chunk_results)
end

"""
    _linear_regions_from_region_map(map_to_regions, mode)

Group full-dimensional polyhedra with the same affine map into
`LinearRegion` objects. This joins pieces only when they meet with codimension
at most one.
"""
function _linear_regions_from_region_map(
        map_to_regions::Dict{Any, Vector{T}},
        mode::LinearRegionsCalculationMode
) where {T}
    linear_regions = LinearRegion{T}[]
    for regions in values(map_to_regions)
        region_components = if length(regions) == 1
            (regions,)
        else
            has_intersection = Dict()
            for (region_1, region_2) in Combinatorics.combinations(regions, 2)
                has_intersection[(
                    region_1, region_2)] = regions_intersect_codimension_le_one(
                    region_1, region_2; mode = mode)
            end

            components(regions, has_intersection)
        end

        for component in region_components
            push!(linear_regions, LinearRegion(convert(Vector{T}, component)))
        end
    end

    isempty(linear_regions) &&
        throw(ArgumentError("No full-dimensional linear regions were found for the rational signomial"))
    return LinearRegions(linear_regions)
end

"""
    linear_regions(q::AbstractVector{<:RationalSignomial}; mode, workers=nothing)

Return the linear regions of `q`.
"""
function linear_regions(
        q::AbstractVector{<:RationalSignomial};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    f = [Q.num for Q in q]
    g = [Q.den for Q in q]
    @assert length(f) == length(g)
    length(f) > 0 ||
        throw(ArgumentError("RationalSignomial vector must have at least one component"))
    any(Q -> length(Q.num) == 0, q) &&
        throw(ArgumentError("RationalSignomial numerator must have at least one monomial"))
    any(Q -> length(Q.den) == 0, q) &&
        throw(ArgumentError("RationalSignomial denominator must have at least one monomial"))

    lin_f = TropicalNN.linear_regions(f; mode = mode, workers = workers)
    lin_g = TropicalNN.linear_regions(g; mode = mode, workers = workers)
    region_type = typeof(lin_f[begin][2])
    if region_type != typeof(lin_g[begin][2])
        throw(ArgumentError(
            "Numerator and denominator regions use incompatible representations: " *
            "$region_type and $(typeof(lin_g[begin][2]))",
        ))
    end
    map_to_regions = Dict{Any, Vector{region_type}}()

    partition_f = _partition_constraint_data(lin_f; mode = mode)
    partition_g = _partition_constraint_data(lin_g; mode = mode)
    intersections = _rational_region_intersections_parallel(
        f,
        g,
        partition_f,
        partition_g,
        mode,
        workers
    )
    for (key, A, b) in intersections
        intersection = make_polyhedron(A, b; mode = mode)
        if haskey(map_to_regions, key)
            push!(map_to_regions[key], intersection)
        else
            map_to_regions[key] = region_type[intersection]
        end
    end

    return _linear_regions_from_region_map(map_to_regions, mode)
end

"""
    linear_regions(q::RationalSignomial; mode, workers=nothing)

Return the linear regions of scalar function `q`.
"""
function linear_regions(
        q::RationalSignomial;
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    length(q.num) > 0 ||
        throw(ArgumentError("RationalSignomial numerator must have at least one monomial"))
    length(q.den) > 0 ||
        throw(ArgumentError("RationalSignomial denominator must have at least one monomial"))

    return linear_regions([q]; mode = mode, workers = workers)
end
