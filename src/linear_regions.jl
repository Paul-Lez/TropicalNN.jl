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

"""
    _canonical_boundary_side(a, rhs)

Return a normalized key for the hyperplane `a*x = rhs` and the side selected
by `a*x <= rhs`. Divide `(a..., -rhs)` by its first nonzero entry, so the first
nonzero key entry is one. The side flag is `true` when the divisor is positive.
Constant inequalities return `nothing`.
"""
function _canonical_boundary_side(a, rhs)
    all(iszero, a) && return nothing
    equation = [a; -rhs]
    pivot_index = findfirst(!iszero, equation)
    pivot = equation[pivot_index]
    key = Tuple(map(equation) do value
        normalized = value / pivot
        return iszero(normalized) ? zero(normalized) : normalized
    end)
    return (key, pivot > 0)
end

"""
    _boundary_sides_from_constraints(A, b)

Return the distinct boundary keys and sides generated by `{x : A*x <= b}`.
"""
function _boundary_sides_from_constraints(A, b)
    boundaries = Tuple{Any, Bool}[]
    seen = Set{Tuple{Any, Bool}}()
    for i in axes(A, 1)
        boundary = _canonical_boundary_side(@view(A[i, :]), b[i])
        boundary === nothing && continue
        boundary in seen && continue
        push!(boundaries, boundary)
        push!(seen, boundary)
    end
    return boundaries
end

"""
    _AbstractCell

Common internal interface for public and internal affine cells.
"""
abstract type _AbstractCell end

@doc raw"""
    Cell(A, b, matrix, offset)

One affine cell of a tropical signomial or rational signomial. The inequalities
`A * x <= b` define the cell. On the cell, the function is
`matrix * x + offset`.
"""
struct Cell{AM, BV, MM, OV} <: _AbstractCell
    A::AM
    b::BV
    matrix::MM
    offset::OV
end

"""
    _Cell(A, b, matrix, offset, data)

Store an affine cell with data used by an internal computation.
The constraint matrix and vector have the same coefficient type.
"""
struct _Cell{
    D,
    T,
    AM <: AbstractMatrix{T},
    BV <: AbstractVector{T},
    MM,
    OV
} <: _AbstractCell
    A::AM
    b::BV
    matrix::MM
    offset::OV
    data::D
end

"""
    Cell(cell::_Cell)

Return the public cell data and omit internal computation data.
"""
Cell(cell::_Cell) = Cell(cell.A, cell.b, cell.matrix, cell.offset)

@doc raw"""
    LinearRegion{C}

One linear region of a tropical signomial or rational signomial. A linear
region can contain more than one affine cell.
"""
struct LinearRegion{C <: Cell}
    cells::Vector{C}
end

Base.length(lr::LinearRegion) = length(lr.cells)
Base.iterate(lr::LinearRegion) = iterate(lr.cells)
Base.iterate(lr::LinearRegion, state) = iterate(lr.cells, state)
Base.getindex(lr::LinearRegion, i::Int) = lr.cells[i]

@doc raw"""
    LinearRegions{C}

Result of `linear_regions`. Each element is a `LinearRegion`.
"""
struct LinearRegions{C <: Cell}
    regions::Vector{LinearRegion{C}}
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

julia> TropicalNN.components(V, D)
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
    @variable(model, epsilon)
    @constraints(model, begin
        A_filtered * x .+ epsilon .<= b_filtered
        # Bound the slack objective.
        epsilon <= 1
    end)
    @objective(model, Max, epsilon)

    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        epsilon_value = value(epsilon)
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
    A_1, b_1 = _region_constraint_data(region_1; mode = mode)
    A_2, b_2 = _region_constraint_data(region_2; mode = mode)
    A = vcat(A_1, A_2)
    b = vcat(b_1, b_2)
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
    make_polyhedron(cell::_AbstractCell; mode)

Construct the selected backend representation of `cell`.
"""
function make_polyhedron(cell::_AbstractCell; mode::LinearRegionsCalculationMode)
    A, b = _region_constraint_data(cell; mode = mode)
    make_polyhedron(A, b; mode = mode)
end

"""
    _linear_region_coefficient_type(mode)

Return the scalar type used to construct linear-region constraints for `mode`.
"""
_linear_region_coefficient_type(::_Oscar) = OSCAR_POLYHEDRON_COEFF_TYPE
_linear_region_coefficient_type(::_HiGHS) = Float64

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
Oscar regions keep exact coefficients. HiGHS regions use `Float64`.
"""
function get_matrix(region::Oscar.Polyhedron; mode::_Oscar)
    return Oscar.halfspace_matrix_pair(Oscar.facets(region)).A
end

function get_matrix(region::_Polyhedra; mode::_HiGHS)
    return region.A
end

"""
    get_matrix(cell::_AbstractCell; mode)

Return the constraint matrix of `cell`.
"""
get_matrix(cell::_AbstractCell; mode::LinearRegionsCalculationMode) = cell.A

"""
    get_vector(region; mode)

Return `b` from the halfspace form `{x : A * x ≤ b}`.
Oscar regions keep exact coefficients. HiGHS regions use `Float64`.
"""
function get_vector(region::Oscar.Polyhedron; mode::_Oscar)
    return Oscar.halfspace_matrix_pair(Oscar.facets(region)).b
end

function get_vector(region::_Polyhedra; mode::_HiGHS)
    return region.b
end

"""
    get_vector(cell::_AbstractCell; mode)

Return the constraint vector of `cell`.
"""
get_vector(cell::_AbstractCell; mode::LinearRegionsCalculationMode) = cell.b

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
    is_feasible(cell::_AbstractCell; mode)

Return whether `cell` is nonempty in the selected backend.
"""
function is_feasible(cell::_AbstractCell; mode::LinearRegionsCalculationMode)
    is_feasible(make_polyhedron(cell; mode = mode); mode = mode)
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

"""
    is_full_dimensional(cell::_AbstractCell; mode)

Return whether `cell` has the ambient dimension in the selected backend.
"""
function is_full_dimensional(cell::_AbstractCell; mode::LinearRegionsCalculationMode)
    is_full_dimensional(make_polyhedron(cell; mode = mode); mode = mode)
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
    A_1, b_1 = _region_constraint_data(region_1; mode = mode)
    A_2, b_2 = _region_constraint_data(region_2; mode = mode)
    A = vcat(A_1, A_2)
    b = vcat(b_1, b_2)
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
    _linear_region_data((signomial, monomial_index, mode))

Construct the constraint data and check full-dimensionality for monomial
`monomial_index` of `signomial`. The input is a tuple so we can directly pass
this to `pmap`.
"""
function _linear_region_data(args)
    signomial, monomial_index, mode = args
    A, b = _linear_region_constraint_data(signomial, monomial_index, mode)
    region = make_polyhedron(A, b; mode = mode)
    return (A, b, is_full_dimensional(region; mode = mode))
end

"""
    _linear_region_data_chunk((signomial, monomial_indices, mode))

Evaluate `_linear_region_data` for a contiguous collection of monomial
indices.
The input is a tuple so we can directly pass this to `pmap`.
"""
function _linear_region_data_chunk(args)
    signomial, monomial_indices, mode = args
    return [_linear_region_data((signomial, monomial_index, mode))
            for monomial_index in monomial_indices]
end

"""
    _linear_region_data_parallel(signomial, mode, workers)

Return dominance-region constraint data for every monomial of `signomial`. When
`workers` is provided, process index chunks with `Distributed.pmap`.
"""
function _linear_region_data_parallel(
        signomial::Signomial,
        mode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool}
)
    monomial_count = length(signomial)
    if workers === nothing || monomial_count <= 1
        return [_linear_region_data((signomial, index, mode))
                for index in Base.eachindex(signomial)]
    end

    _assert_tropicalnn_loaded(workers)
    chunks = _index_chunks(monomial_count, length(Distributed.workers(workers)))
    chunk_results = Distributed.pmap(
        _linear_region_data_chunk,
        workers,
        [(signomial, chunk, mode) for chunk in chunks]
    )
    return Base.reduce(vcat, chunk_results)
end

"""
    linear_regions(f::Signomial; mode, workers=nothing)

Return the linear regions of `f`.
"""
function linear_regions(
        f::Signomial;
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    return linear_regions([f]; mode = mode, workers = workers)
end

"""
    _intersect_linear_region_partitions(partitions; mode)

Intersect the region partitions one at a time. Discard a partial intersection
that is not full dimensional before adding a region from the next partition.
"""
function _intersect_linear_region_partitions(
        partitions;
        mode::LinearRegionsCalculationMode
)
    regions = [((index,), region) for (index, region) in first(partitions)]

    for partition in Iterators.drop(partitions, 1)
        next_regions = []
        for (index, candidate_region) in partition
            for (indices, partial_region) in regions
                region = region_intersection(
                    partial_region,
                    candidate_region;
                    mode = mode
                )
                if is_full_dimensional(region; mode = mode)
                    push!(next_regions, ((indices..., index), region))
                end
            end
        end
        regions = next_regions
        isempty(regions) && break
    end

    return regions
end

"""
    _signomial_region_partition(signomials; mode, workers=nothing)

Return the common dominance-cell partition of the signomial vector `signomials`.
Each cell stores its active monomial indices as internal data.
"""
function _signomial_region_partition(
        signomials::AbstractVector{<:Signomial};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    isempty(signomials) && return _Cell[]

    dominance_partitions = map(signomials) do signomial
        region_data = _linear_region_data_parallel(signomial, mode, workers)
        return [(monomial_index, make_polyhedron(data[1], data[2]; mode = mode))
                for (monomial_index, data) in pairs(region_data) if data[3]]
    end

    partition = _intersect_linear_region_partitions(dominance_partitions; mode = mode)
    return map(partition) do (dominance_indices, region)
        A, b = _region_constraint_data(region; mode = mode)
        affine_key = [(Rational(get_coeff(signomial, index)),
                          collect(get_exp(signomial, index)))
                      for (signomial, index) in zip(signomials, dominance_indices)]
        matrix, offset = _affine_formula_from_linear_map_key(affine_key)
        return _Cell(A, b, matrix, offset, dominance_indices)
    end
end

"""
    linear_regions(f::AbstractVector{<:Signomial}; mode, workers=nothing)

Return the linear regions of `f`.
"""
function linear_regions(
        f::AbstractVector{<:Signomial};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    partition = _signomial_region_partition(f; mode = mode, workers = workers)
    regions = LinearRegion{Cell}[]
    for cell in partition
        push!(regions, LinearRegion(Cell[Cell(cell)]))
    end
    return LinearRegions(regions)
end

"""
    _linear_map_key(numerator, denominator, numerator_index, denominator_index)

Return a hashable representation of the affine map obtained by subtracting
monomial `denominator_index` of `denominator` from monomial `numerator_index` of
`numerator`.
"""
function _linear_map_key(
        numerator::Signomial,
        denominator::Signomial,
        numerator_index,
        denominator_index
)
    coeff = Rational(get_coeff(numerator, numerator_index)) -
            Rational(get_coeff(denominator, denominator_index))
    exp = collect(get_exp(numerator, numerator_index)) -
          collect(get_exp(denominator, denominator_index))
    return (coeff, exp)
end

function _linear_map_key(f::Vector{<:Signomial}, g::Vector{<:Signomial}, idxf, idxg)
    # Each vector has one entry for each output coordinate.
    @assert length(f) == length(g) == length(idxf) == length(idxg)
    return map(i -> _linear_map_key(f[i], g[i], idxf[i], idxg[i]), Base.eachindex(idxf))
end

"""
    _affine_formula_from_linear_map_key(key)

Convert a rational-subdivision key into the matrix and offset of its affine
map. The key has one `(offset, coefficients)` pair for each output coordinate.
The coefficients are the corresponding row of the affine matrix.
"""
function _affine_formula_from_linear_map_key(key)
    isempty(key) && throw(ArgumentError("An affine map must have at least one output"))
    rows = [permutedims(collect(component[2])) for component in key]
    matrix = reduce(vcat, rows)
    offset = [component[1] for component in key]
    return matrix, offset
end

"""
    _affine_map_key(matrix, offset)

Return an immutable, globally comparable key for an affine map.
"""
function _affine_map_key(matrix, offset)
    return (size(matrix), Tuple(vec(matrix)), Tuple(offset))
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
    A = vcat(facets.A, affine_hull.A, -affine_hull.A)
    b = vcat(facets.b, affine_hull.b, -affine_hull.b)
    return (
        _constraint_matrix(OSCAR_POLYHEDRON_COEFF_TYPE, A),
        _constraint_vector(OSCAR_POLYHEDRON_COEFF_TYPE, b)
    )
end

function _region_constraint_data(region::_Polyhedra; mode::_HiGHS)
    return (region.A, region.b)
end

"""
    _region_constraint_data(cell::_AbstractCell; mode)

Return the stored halfspace matrix and vector of `cell`.
"""
function _region_constraint_data(
        cell::_AbstractCell;
        mode::LinearRegionsCalculationMode
)
    return (cell.A, cell.b)
end

"""
    _rational_region_intersections_chunk((numerator_data, denominator_data,
                                          candidate_pairs, mode))

Test the requested numerator/denominator partition pairs for full-dimensional
intersection. Return an internal cell for each accepted intersection.
The input is a tuple so we can directly pass this to `pmap`.
"""
function _rational_region_intersections_chunk(args)
    numerator_data, denominator_data, candidate_pairs, mode = args
    cells = _Cell[]

    for (numerator_cell_index, denominator_cell_index) in candidate_pairs
        numerator_cell = numerator_data[numerator_cell_index]
        denominator_cell = denominator_data[denominator_cell_index]
        intersection_matrix = vcat(numerator_cell.A, denominator_cell.A)
        intersection_vector = vcat(numerator_cell.b, denominator_cell.b)
        region = make_polyhedron(intersection_matrix, intersection_vector; mode = mode)
        if is_full_dimensional(region; mode = mode)
            push!(
                cells,
                _Cell(
                    intersection_matrix,
                    intersection_vector,
                    numerator_cell.matrix - denominator_cell.matrix,
                    numerator_cell.offset - denominator_cell.offset,
                    _boundary_sides_from_constraints(intersection_matrix, intersection_vector)
                )
            )
        end
    end

    return cells
end

"""
    _rational_region_intersections_parallel(numerator_data, denominator_data,
                                            mode, workers)

Test every numerator/denominator partition pair for full-dimensional
intersection. Use `workers` to evaluate pair chunks in parallel when supplied.
"""
function _rational_region_intersections_parallel(
        numerator_data::AbstractVector{<:_Cell},
        denominator_data::AbstractVector{<:_Cell},
        mode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool}
)
    candidate_pairs = Tuple{Int, Int}[]
    for numerator_cell_index in Base.eachindex(numerator_data)
        for denominator_cell_index in Base.eachindex(denominator_data)
            push!(candidate_pairs, (numerator_cell_index, denominator_cell_index))
        end
    end

    isempty(candidate_pairs) && return _Cell[]
    if workers === nothing || length(candidate_pairs) <= 1
        return _rational_region_intersections_chunk(
            (numerator_data, denominator_data, candidate_pairs, mode)
        )
    end

    _assert_tropicalnn_loaded(workers)
    chunks = _index_chunks(
        length(candidate_pairs), length(Distributed.workers(workers)))
    pair_chunks = [candidate_pairs[chunk] for chunk in chunks]
    chunk_results = Distributed.pmap(
        _rational_region_intersections_chunk,
        workers,
        [(numerator_data, denominator_data, pair_chunk, mode)
         for pair_chunk in pair_chunks]
    )
    return Base.reduce(vcat, chunk_results)
end

"""
    _cell_components(cells; mode)

Return index sets for components formed by codimension-at-most-one adjacency.
"""
function _cell_components(
        cells::AbstractVector{<:_Cell};
        mode::LinearRegionsCalculationMode
)
    length(cells) <= 1 && return [collect(Base.eachindex(cells))]

    graph = Graphs.SimpleGraph(length(cells))
    for (left_index, right_index) in Combinatorics.combinations(collect(Base.eachindex(cells)), 2)
        if regions_intersect_codimension_le_one(
            cells[left_index], cells[right_index]; mode = mode)
            Graphs.add_edge!(graph, left_index, right_index)
        end
    end
    return Graphs.connected_components(graph)
end

"""
    _group_cells(cells; mode)

Group cells by affine-map equality and split each group into connected linear
regions. Return the constituent cells, the linear regions, the number of
affine-map groups, and the number of connected components.
"""
function _group_cells(
        cells::AbstractVector{C};
        mode::LinearRegionsCalculationMode
) where {C <: _Cell}
    isempty(cells) && throw(ArgumentError(
        "No full-dimensional linear regions were found for the rational signomial"
    ))

    map_to_indices = Dict{Any, Vector{Int}}()
    for (index, cell) in pairs(cells)
        key = _affine_map_key(cell.matrix, cell.offset)
        push!(get!(map_to_indices, key, Int[]), index)
    end

    grouped_cells = C[]
    linear_regions = LinearRegion{Cell}[]
    component_count = 0
    for indices in values(map_to_indices)
        affine_cells = cells[indices]
        for component_indices in _cell_components(affine_cells; mode = mode)
            component_count += 1
            component = affine_cells[component_indices]
            append!(grouped_cells, component)
            push!(linear_regions, LinearRegion(Cell[Cell(cell) for cell in component]))
        end
    end

    return (
        grouped_cells,
        LinearRegions(linear_regions),
        length(map_to_indices),
        component_count
    )
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

    numerator = _signomial_region_partition(f; mode = mode, workers = workers)
    denominator = _signomial_region_partition(g; mode = mode, workers = workers)
    cells = _rational_region_intersections_parallel(
        numerator,
        denominator,
        mode,
        workers
    )
    _, regions, _, _ = _group_cells(cells; mode = mode)
    return regions
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
