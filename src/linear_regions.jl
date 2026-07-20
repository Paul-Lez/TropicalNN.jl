# Shared implementation for computing linear regions of tropical Puiseux
# polynomials and rational functions.

const OSCAR_POLYHEDRON_COEFF_TYPE = Rational{BigInt}
const HIGHS_DEFAULT_TOL = 1e-6
const HIGHS_DEFAULT_SOLVER = "choose"

function _normalise_worker_ids(workers)
    workers === nothing && return nothing

    worker_ids = unique(Int[worker for worker in workers])
    filter!(pid -> pid != Distributed.myid(), worker_ids)
    isempty(worker_ids) && return nothing

    active_processes = Distributed.procs()
    inactive = setdiff(worker_ids, active_processes)
    if !isempty(inactive)
        throw(ArgumentError("Inactive Julia worker process ids: $(join(inactive, ", "))"))
    end
    return worker_ids
end

function _ensure_tropicalnn_loaded(worker_ids)
    for pid in worker_ids
        try
            Distributed.remotecall_wait(Core.eval, pid, Main, :(using TropicalNN))
        catch err
            throw(ArgumentError(
                "Worker $pid could not load TropicalNN. Start workers with the same " *
                "project, for example addprocs(...; exeflags=[\"--project=.\"]), " *
                "before passing worker ids. Original error: $err",
            ))
        end
    end
    return nothing
end

function _index_chunks(n::Int, nworkers::Int)
    n <= 0 && return UnitRange{Int}[]

    nchunks = min(n, 4 * max(1, nworkers))
    chunk_size = cld(n, nchunks)
    return [start:min(start + chunk_size - 1, n) for start in 1:chunk_size:n]
end

"""
    LinearRegionsCalculationMode

Abstract supertype for backend selectors used by the mode-based linear-region
API.
"""
abstract type LinearRegionsCalculationMode end

struct _Oscar <: LinearRegionsCalculationMode end

Base.@kwdef struct _HiGHS <: LinearRegionsCalculationMode
    tol::Float64 = HIGHS_DEFAULT_TOL
    solver::String = HIGHS_DEFAULT_SOLVER
    threads::Union{Nothing, Int} = nothing
end

"""
    OscarMode()

Use Oscar polyhedra and exact rational arithmetic for linear-region calculations.
"""
const OscarMode = _Oscar

"""
    HiGHSMode(; tol=HIGHS_DEFAULT_TOL, solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Use JuMP/HiGHS LP checks and floating-point constraint matrices for
linear-region calculations. `tol` controls full-dimensionality checks.
`threads` optionally sets the HiGHS thread count.
"""
const HiGHSMode = _HiGHS

struct _Polyhedra
    A::Matrix{Float64}
    b::Vector{Float64}
end

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

function _constraint_vector(::Type{T}, values) where {T}
    return T[_constraint_scalar(T, value) for value in values]
end

function _constraint_matrix(::Type{T}, values) where {T}
    return Matrix{T}(map(value -> _constraint_scalar(T, value), values))
end

function _linear_region_constraints(
        f::Signomial,
        i,
        ::Type{T};
        include_self::Bool = false,
        competitors = Base.eachindex(f),
        empty_rows::Int = 1
) where {T}
    indices = [j for j in competitors if include_self || j != i]
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

The return type of `linear_regions`. Holds all linear regions
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

function _components_graph(V, D)
    # TODO: get rid of this and create the graph on the fly instead.
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

function n_components(V, D)
    return length(Graphs.connected_components(_components_graph(V, D)))
end

@doc raw"""
    components(V::Vector{T}, D::Dict{Tuple{T, T}, Bool})

Outputs an array representing the connected components of the graph given by the vertices V and the edges D
(more precisely, the edges are given by the keys of D whose entries are "true").
True edge endpoints are expected to belong to `V`.

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

Check if polyhedron `{x : Ax <= b}` is empty via LP feasibility using HiGHS.
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
    # Here we optimise with no objective, which means that
    # JuMP will stop as soon as a solution is found.
    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        return false
    elseif status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
        return true
    end
    throw(ErrorException("HiGHS feasibility check ended with unexpected status $status"))
end

# Remove constant inequalities from a linear program (assuming they are satisfied)
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

Check if polyhedron `{x : Ax <= b}` is full dimensional via LP using HiGHS.
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
    # infeasible polyhedron is not full-dimensional
    if filtered === nothing
        return false
    end
    A_filtered, b_filtered = filtered

    # If there are no non-trivial rows then the polyhedron is all of R^n
    if isempty(b_filtered)
        return true
    end

    model = create_highs_model(; solver = solver, threads = threads)

    @variable(model, x[1:n])
    @variable(model, ε)
    @constraints(model, begin
        A_filtered * x .+ ε .<= b_filtered
        # Make sure the problem is bounded
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

Check if the intersection of two polyhedra is full dimensional via LP using HiGHS.
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

# returns true if the i-th inequality is redundant.
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
    # We maximise a_i^T x - b_i to check if the i-th inequality is redundant.
    @objective(model, Max, s)

    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        s_value = value(s)
        return !(s_value > tol)
    elseif status == MOI.INFEASIBLE_OR_UNBOUNDED || status == MOI.DUAL_INFEASIBLE
        # The problem is unbounded, which means that the i-th inequality is not equality
        return false
    else
        # The problem should always be feasible, so any other status is unexpected.
        throw(ErrorException("HiGHS full-dimensionality check ended with unexpected status $status"))
    end
end

# Check if the polyhedron defined by Ax <= b is codimension one via LP using HiGHS.
function _highs_codimension_le_one(A::Matrix{Float64},
        b::Vector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = HIGHS_DEFAULT_SOLVER,
        threads = nothing
    )

    # Remove constant inequalities from the LP.
    filtered = filter_lp(A, b)
    # If the polyhedron is infeasible, it cannot be of codimension at most one
    if filtered === nothing
        return false
    end
    A_filtered, b_filtered = filtered

    if highs_is_empty(A_filtered, b_filtered; solver = solver, threads = threads)
        return false
    end

    if highs_is_full_dimensional(A_filtered, b_filtered; tol = tol, solver = solver, threads = threads)
        return true
    end

    m, n = size(A)

    redundantIdx = []

    for i in 1:size(A_filtered, 1)
        # if there are at least two independent redundant inequalities, then the polyhedron must have at least codimension two
        if highs_check_implicit_equality(A_filtered, b_filtered, i; tol = tol, solver = solver, threads = threads)
            push!(redundantIdx, i)
        end
        if length(redundantIdx) > 1
            # We check independence by computing the rank of the matrix of redundant inequalities.
            A_redundant = A_filtered[redundantIdx, :]
            if rank(A_redundant) > 1
                return false
            end
        end
    end
    return true
end

# check that the codimension of the polyhedron is at most one
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

function regions_intersect_codimension_le_one(region_1, region_2; mode::LinearRegionsCalculationMode)
    A = vcat(get_matrix(region_1; mode = mode), get_matrix(region_2; mode = mode))
    b = vcat(get_vector(region_1; mode = mode), get_vector(region_2; mode = mode))
    return codimension_le_one(A, b; mode = mode)
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

function _linear_region_coefficient_type(mode::LinearRegionsCalculationMode)
    if mode isa _Oscar
        return OSCAR_POLYHEDRON_COEFF_TYPE
    else
        return Float64
    end
end

function _linear_region_empty_rows(mode::LinearRegionsCalculationMode)
    if mode isa _HiGHS
        return 0
    else
        return 1
    end
end

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
        competitors = competitors,
        empty_rows = _linear_region_empty_rows(mode),
    )
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
    return !highs_is_empty(region.A, region.b; solver = mode.solver, threads = mode.threads)
end

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

Outputs the polyhedron corresponding to points where f is given by the
linear map corresponding to the i-th monomial of f, using the selected backend
mode.

`competitors` restricts the monomials that the i-th monomial is compared
against. Callers may drop monomials already known to be redundant (i.e. whose
dominance polyhedron is not full-dimensional): removing such a monomial does
not change `f` as a function, so the resulting polyhedron is the same set
described by fewer inequalities.
"""
function polyhedron(
        f::Signomial,
        i,
        mode::LinearRegionsCalculationMode;
        competitors = Base.eachindex(f)
)
    A, b = _linear_region_constraint_data(f, i, mode; competitors = competitors)
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

function _linear_region_data(args)
    f, i, mode = args
    A, b = _linear_region_constraint_data(f, i, mode)
    region = make_polyhedron(A, b; mode = mode)
    return (A, b, is_feasible(region; mode = mode))
end

function _linear_region_data_chunk(args)
    f, inds, mode = args
    return [_linear_region_data((f, i, mode)) for i in inds]
end

function _linear_region_data_parallel(f::Signomial, mode, worker_ids)
    n = length(f)
    if worker_ids === nothing || n <= 1
        return [_linear_region_data((f, i, mode)) for i in Base.eachindex(f)]
    end

    _ensure_tropicalnn_loaded(worker_ids)
    chunks = _index_chunks(n, length(worker_ids))
    chunk_results = Distributed.pmap(
        _linear_region_data_chunk,
        Distributed.WorkerPool(worker_ids),
        [(f, chunk, mode) for chunk in chunks],
    )
    return Base.reduce(vcat, chunk_results)
end

function _linear_region_from_data(data, mode::LinearRegionsCalculationMode)
    A, b, feasible = data
    return (make_polyhedron(A, b; mode = mode), feasible)
end

"""
    enum_linear_regions_general(f::Signomial; mode, workers=nothing)

Compute all linear-region candidates for signomial `f` using the selected
backend mode. If `workers` is supplied, candidate construction and feasibility
checks run on those Julia worker processes.

Returns a vector of `(region, is_feasible)` tuples indexed by the monomials of
`f`.
"""
function enum_linear_regions_general(
        f::Signomial;
        mode::LinearRegionsCalculationMode,
        workers = nothing
)
    worker_ids = _normalise_worker_ids(workers)
    region_data = _linear_region_data_parallel(f, mode, worker_ids)
    return [_linear_region_from_data(data, mode) for data in region_data]
end

"""
    linear_regions(f::Signomial; mode, workers=nothing)

Compute all linear-region candidates for signomial `f` using the selected
backend mode.

Returns a vector of `(index, region)` tuples indexed by the monomials of `f`.
Feasibility of each candidate region can be checked with [`is_feasible`](@ref).
"""
function linear_regions(
        f::Signomial;
        mode::LinearRegionsCalculationMode,
        workers = nothing
)
    worker_ids = _normalise_worker_ids(workers)
    region_data = _linear_region_data_parallel(f, mode, worker_ids)
    return [(i, make_polyhedron(data[1], data[2]; mode = mode))
            for (i, data) in pairs(region_data)]
end

# Computes the linear regions of a vector of signomials, returning a vector of tuples
# (index_tuple, region) where index_tuple is a tuple of indices into the vector of signomials 
# and region is the intersection of the corresponding regions.
function linear_regions(
        f::AbstractVector{<:Signomial};
        mode::LinearRegionsCalculationMode,
        workers = nothing
)
    regions = []
    linear_regions_vec = map(g -> linear_regions(g; mode = mode, workers = workers), f)
    for idx in Iterators.product((Base.eachindex(g) for g in linear_regions_vec)...)
        regionsIdx = map(i -> begin
            idxj = idx[i]
            @assert idxj <= length(linear_regions_vec[i])
            u = linear_regions_vec[i][idx[i]][2]
        end, 
        Base.eachindex(f))
        regionIdx = Base.reduce((r1, r2) -> region_intersection(r1, r2; mode = mode), regionsIdx)
        if is_feasible(regionIdx; mode = mode)
            push!(regions, (idx, regionIdx))
        end
    end

    return regions
end

function _linear_map_key(f::Signomial, g::Signomial, i, j)
    coeff = Rational(get_coeff(f, i)) - Rational(get_coeff(g, j))
    exp = collect(get_exp(f, i)) - collect(get_exp(g, j))
    return (coeff, exp)
end

function _linear_map_key(f::Vector{<:Signomial}, g::Vector{<:Signomial}, idxf, idxg)
    # All vectors in the inputs are assumed to have the same lengths
    @assert length(f) == length(g) == length(idxf) == length(idxg)
    return map(i -> _linear_map_key(f[i], g[i], idxf[i], idxg[i]), Base.eachindex(idxf))
end

function _rational_region_intersections_chunk(args)
    f, g, lin_f, lin_g, pairs, mode = args
    intersections = Vector{Tuple{Any, Any, Any}}()

    for (i, j) in pairs
        A = vcat(lin_f[i][1], lin_g[j][1])
        b = vcat(lin_f[i][2], lin_g[j][2])
        intersection = make_polyhedron(A, b; mode = mode)
        if is_full_dimensional(intersection; mode = mode)
            push!(intersections, (_linear_map_key(f, g, i, j), A, b))
        end
    end

    return intersections
end

function _rational_region_intersections_parallel(f, g, lin_f, lin_g, mode, worker_ids)
    pairs = Tuple{Int, Int}[]
    for i in Base.eachindex(f)
        for j in Base.eachindex(g)
            if lin_f[i][3] && lin_g[j][3]
                push!(pairs, (i, j))
            end
        end
    end

    isempty(pairs) && return Tuple{Any, Any, Any}[]
    if worker_ids === nothing || length(pairs) <= 1
        return _rational_region_intersections_chunk((f, g, lin_f, lin_g, pairs, mode))
    end

    _ensure_tropicalnn_loaded(worker_ids)
    chunks = _index_chunks(length(pairs), length(worker_ids))
    pair_chunks = [pairs[chunk] for chunk in chunks]
    chunk_results = Distributed.pmap(
        _rational_region_intersections_chunk,
        Distributed.WorkerPool(worker_ids),
        [(f, g, lin_f, lin_g, pair_chunk, mode) for pair_chunk in pair_chunks],
    )
    return Base.reduce(vcat, chunk_results)
end

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
                has_intersection[(region_1, region_2)] =
                    regions_intersect_codimension_le_one(region_1, region_2; mode = mode)
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

function _linear_regions_rat_distributed(q::RationalSignomial, mode, worker_ids)
    f = q.num
    g = q.den

    lin_f = _linear_region_data_parallel(f, mode, worker_ids)
    lin_g = _linear_region_data_parallel(g, mode, worker_ids)
    region_type = typeof(make_polyhedron(lin_f[begin][1], lin_f[begin][2]; mode = mode))
    map_to_regions = Dict{Any, Vector{region_type}}()

    intersections = _rational_region_intersections_parallel(f, g, lin_f, lin_g, mode, worker_ids)
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
    linear_regions(q::AbstractVector{<:RationalSignomial}; mode, workers=nothing)

Compute the linear regions of a vector-valued tropical Puiseux rational function
using the algorithm indicated by `mode`.
"""
function linear_regions(
        q::AbstractVector{<:RationalSignomial};
        mode::LinearRegionsCalculationMode,
        workers = nothing
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

    for (idxf, regionf) in lin_f
        for (idxg, regiong) in lin_g
            intersection = region_intersection(regionf, regiong; mode = mode)
            if is_full_dimensional(intersection; mode = mode)
                key = _linear_map_key(f, g, idxf, idxg)
                if haskey(map_to_regions, key)
                    push!(map_to_regions[key], intersection)
                else
                    map_to_regions[key] = region_type[intersection]
                end
            end
        end
    end

    return _linear_regions_from_region_map(map_to_regions, mode)
end


"""
    linear_regions(q::RationalSignomial; mode, workers=nothing)

Compute the linear regions of a tropical Puiseux rational function using the
algorithm indicated by `mode`. If `workers` is `nothing`, this runs locally;
otherwise independent candidate and intersection checks run on those Julia
worker processes.
"""
function linear_regions(
        q::RationalSignomial;
        mode::LinearRegionsCalculationMode,
        workers = nothing
)
    length(q.num) > 0 ||
        throw(ArgumentError("RationalSignomial numerator must have at least one monomial"))
    length(q.den) > 0 ||
        throw(ArgumentError("RationalSignomial denominator must have at least one monomial"))

    worker_ids = _normalise_worker_ids(workers)
    if worker_ids !== nothing
        return _linear_regions_rat_distributed(q, mode, worker_ids)
    end
    return linear_regions([q]; mode = mode)
end
