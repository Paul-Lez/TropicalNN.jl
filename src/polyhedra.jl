# Backend infrastructure for polyhedral computations.

const OSCAR_POLYHEDRON_COEFF_TYPE = Rational{BigInt}
const HIGHS_DEFAULT_TOL = 1e-6
const HIGHS_DEFAULT_SOLVER = "choose"

"""
    _validate_highs_tolerance(tol)

Check that `tol` is greater than zero.
"""
function _validate_highs_tolerance(tol)
    tol > 0 || throw(ArgumentError("tol must be positive, got $tol"))
    return nothing
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
    highs_is_empty(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}; solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Check if polyhedron `{x : Ax <= b}` is empty by solving a linear program using HiGHS.
"""
function highs_is_empty(
        A::AbstractMatrix{Float64},
        b::AbstractVector{Float64};
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
function filter_lp(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)

    zero_rows = [i for i in 1:m if all(iszero, A[i, :])]
    any(i -> b[i] < 0, zero_rows) && return nothing
    non_trivial_rows = [i for i in 1:m if !(i in zero_rows)]

    A_filtered = A[non_trivial_rows, :]
    b_filtered = b[non_trivial_rows]

    return A_filtered, b_filtered
end

"""
    highs_is_full_dimensional(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}; tol=HIGHS_DEFAULT_TOL, solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Check if polyhedron `{x : Ax <= b}` is full dimensional by solving a linear program using HiGHS.
"""
function highs_is_full_dimensional(
        A::AbstractMatrix{Float64},
        b::AbstractVector{Float64};
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
    highs_intersect_is_full_dimensional(A1::AbstractMatrix{Float64}, b1::AbstractVector{Float64},
                                        A2::AbstractMatrix{Float64}, b2::AbstractVector{Float64};
                                        tol=HIGHS_DEFAULT_TOL, solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Check if the intersection of two polyhedra is full dimensional via by solving linear programs using HiGHS.
"""
function highs_intersect_is_full_dimensional(
        A1::AbstractMatrix{Float64}, b1::AbstractVector{Float64},
        A2::AbstractMatrix{Float64}, b2::AbstractVector{Float64};
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
        A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, i::Int;
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
function _highs_codimension_le_one(A::AbstractMatrix{Float64},
        b::AbstractVector{Float64};
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
            if LinearAlgebra.rank(A_redundant) > 1
                return false
            end
        end
    end
    return true
end

"""
    _hyperplanes_coincide(left_row, left_rhs, right_row, right_rhs, mode)

Return whether two exact affine equations have the same zero set.
"""
function _hyperplanes_coincide(left_row, left_rhs, right_row, right_rhs, ::_Oscar)
    pivot_index = findfirst(!iszero, left_row)
    pivot_index === nothing && return false
    left_pivot = left_row[pivot_index]
    right_pivot = right_row[pivot_index]
    iszero(right_pivot) && return false

    for index in eachindex(left_row, right_row)
        left_row[index] * right_pivot == right_row[index] * left_pivot || return false
    end
    return left_rhs * right_pivot == right_rhs * left_pivot
end

"""
    _hyperplanes_coincide(left_row, left_rhs, right_row, right_rhs, mode)

Return whether two Float64 affine equations represent the same hyperplane.
"""
function _hyperplanes_coincide(left_row, left_rhs, right_row, right_rhs, ::_HiGHS)
    pivot_index = argmax(abs.(left_row))
    left_pivot = left_row[pivot_index]
    right_pivot = right_row[pivot_index]
    (iszero(left_pivot) || iszero(right_pivot)) && return false

    left_scale = inv(left_pivot)
    right_scale = inv(right_pivot)
    for index in eachindex(left_row, right_row)
        isapprox(
            left_row[index] * left_scale,
            right_row[index] * right_scale;
            atol = 1.0e-9,
            rtol = 1.0e-9
        ) || return false
    end
    return isapprox(
        left_rhs * left_scale,
        right_rhs * right_scale;
        atol = 1.0e-9,
        rtol = 1.0e-9
    )
end

"""
    _dominance_transition_indices(signomial, monomial_index, mode)

Return the monomials whose dominance regions can share a facet with
`monomial_index`. Keep all comparisons on a retained facet, including
proportional and reversed representations.
"""
function _dominance_transition_indices(
        signomial::Signomial,
        monomial_index::Int,
        mode::LinearRegionsCalculationMode
)
    A, b = _linear_region_constraint_data(signomial, monomial_index, mode)
    isempty(b) && return Int[]

    arithmetic = mode isa _Oscar ? :exact : :float
    region = Polyhedra.polyhedron(
        Polyhedra.hrep(A, b),
        CDDLib.Library(arithmetic)
    )
    redundant_rows = CDDLib.gethredundantindices(region)
    facet_rows = [index for index in eachindex(b) if !(index in redundant_rows)]
    competitor_indices = [index for index in Base.eachindex(signomial)
                          if index != monomial_index]

    transition_indices = Int[]
    for row_index in eachindex(b)
        any(facet_rows) do facet_index
            _hyperplanes_coincide(
                @view(A[row_index, :]),
                b[row_index],
                @view(A[facet_index, :]),
                b[facet_index],
                mode
            )
        end || continue
        push!(transition_indices, competitor_indices[row_index])
    end
    return transition_indices
end

"""
    _highs_cells_share_facet(left, right, transition, boundary_sources, mode)

Test whether two HiGHS cells share a facet on the selected monomial equality.
"""
function _highs_cells_share_facet(
        left::_Cell,
        right::_Cell,
        transition::_MonomialTransition,
        boundary_sources,
        mode::_HiGHS
)
    signomial = boundary_sources[transition.source_id][transition.component_id]
    candidate_matrix, candidate_vector = _linear_region_constraints(
        signomial,
        transition.lower_index,
        Float64;
        competitors = (transition.upper_index,)
    )
    candidate_row = @view candidate_matrix[1, :]
    candidate_rhs = candidate_vector[1]
    candidate_scale = LinearAlgebra.norm(candidate_row)
    iszero(candidate_scale) && return false
    normal = candidate_row ./ candidate_scale
    rhs = candidate_rhs / candidate_scale
    isfinite(rhs) || return false

    # Constrain the common face to the candidate monomial equality.
    model = create_highs_model(; solver = mode.solver, threads = mode.threads)
    @variable(model, x[axes(left.A, 2)])
    @variable(model, epsilon)
    @constraint(model, LinearAlgebra.dot(normal, x) == rhs)
    @constraint(model, epsilon <= 1)

    # Require positive slack away from the candidate hyperplane.
    for (A, b) in ((left.A, left.b), (right.A, right.b))
        for row_index in axes(A, 1)
            row = @view A[row_index, :]
            all(iszero, row) && continue
            _hyperplanes_coincide(
                row,
                b[row_index],
                candidate_row,
                candidate_rhs,
                mode
            ) && continue
            row_norm = LinearAlgebra.norm(row)
            @constraint(model,
                LinearAlgebra.dot(row, x) + row_norm * epsilon <= b[row_index])
        end
    end

    @objective(model, Max, epsilon)
    optimize!(model)
    status = termination_status(model)
    if status == MOI.OPTIMAL
        return value(epsilon) > mode.tol
    elseif status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
        return false
    end
    throw(ErrorException("HiGHS facet check ended with unexpected status $status"))
end

"""
    _cells_share_facet(left, right, transitions, boundary_sources, mode)

Test whether two cells share a facet with the selected geometry backend.
"""
function _cells_share_facet(
        left::_Cell,
        right::_Cell,
        transitions,
        boundary_sources,
        mode::_HiGHS
)
    return _highs_cells_share_facet(
        left,
        right,
        first(transitions),
        boundary_sources,
        mode
    )
end

function _cells_share_facet(
        left::_Cell,
        right::_Cell,
        _,
        _,
        mode::LinearRegionsCalculationMode
)
    return regions_intersect_codimension_le_one(left, right; mode = mode)
end

"""
    codimension_le_one(A, b; mode)

Return whether `{x : Ax ≤ b}` has codimension at most one.
"""
function codimension_le_one(A, b; mode::LinearRegionsCalculationMode)
    if mode isa _Oscar
        poly = make_polyhedron(A, b; mode = mode)
        return Oscar.is_feasible(poly) && Oscar.codim(poly) <= 1
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
    get_matrix(cell::_AbstractCell; mode)

Return the constraint matrix of `cell`.
"""
get_matrix(cell::_AbstractCell; mode::LinearRegionsCalculationMode) = cell.A

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
