# Backend infrastructure for polyhedral computations.

const OSCAR_POLYHEDRON_COEFF_TYPE = Rational{BigInt}

"""
    HIGHS_DEFAULT_TOL

`HIGHS_DEFAULT_TOL` is the default distance tolerance for HiGHS region checks,
in input-space units.
"""
const HIGHS_DEFAULT_TOL = 1e-6
const HIGHS_DEFAULT_SOLVER = "choose"

"""
    _validate_highs_tolerance(tol)

Throw an `ArgumentError` if `tol` is not finite or if `tol ≤ 0`.
"""
function _validate_highs_tolerance(tol)
    isfinite(tol) && tol > 0 ||
        throw(ArgumentError("tol must be finite and positive, got $tol"))
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

Store the options for linear-region checks with HiGHS.
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

Use JuMP and HiGHS with `Float64` constraints. `tol` is the distance tolerance
for region checks, in input-space units. A region is full dimensional only if
its Chebyshev inradius is greater than `tol`. `threads` sets the optional HiGHS
thread count.
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

function _constraint_scalar(::Type{Float64}, x::_FLOAT_TROPICAL_COEFF)
    return _constraint_scalar(Float64, TropicalNumbers.content(x))
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

function _constraint_scalar(
        ::Type{OSCAR_POLYHEDRON_COEFF_TYPE},
        x::_FLOAT_TROPICAL_COEFF
)
    return _constraint_scalar(
        OSCAR_POLYHEDRON_COEFF_TYPE,
        TropicalNumbers.content(x)
    )
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
    create_glpk_model()

Create a silent GLPK model for LP checks that HiGHS cannot resolve.
"""
function create_glpk_model()
    model = Model(GLPK.Optimizer)
    set_silent(model)
    return model
end

"""
    _solve_with_glpk_fallback(solve_lp, highs_model, check_name)

Run an LP check with HiGHS. Retry it with GLPK when HiGHS returns a status
that the check cannot classify.
"""
function _solve_with_glpk_fallback(solve_lp, highs_model, check_name)
    result, highs_status = solve_lp(highs_model)
    result !== nothing && return result

    result, glpk_status = solve_lp(create_glpk_model())
    result !== nothing && return result

    throw(ErrorException(
        "GLPK $check_name ended with unexpected status $glpk_status " *
        "after HiGHS ended with unexpected status $highs_status"
    ))
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

    filtered = filter_lp(A, b)
    filtered === nothing && return true
    A_filtered, b_filtered = filtered
    isempty(b_filtered) && return false

    A_normalized, b_normalized = normalize_lp(A_filtered, b_filtered)

    model = create_highs_model(; solver = solver, threads = threads)

    @variable(model, x[1:n])
    @constraint(model, A_normalized * x .<= b_normalized)
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
    normalize_lp(A, b)

Return an equivalent halfspace system with unit Euclidean row norms.
"""
function normalize_lp(A::AbstractMatrix{Float64}, b::AbstractVector{Float64})
    scales = [LinearAlgebra.norm(row) for row in eachrow(A)]
    return A ./ scales, b ./ scales
end

"""
    highs_is_full_dimensional(A::AbstractMatrix{Float64}, b::AbstractVector{Float64}; tol=HIGHS_DEFAULT_TOL, solver=HIGHS_DEFAULT_SOLVER, threads=nothing)

Return `true` if `{x : Ax ≤ b}` has a Chebyshev inradius greater than `tol`.
The tolerance is in input-space units.
"""
function highs_is_full_dimensional(
        A::AbstractMatrix{Float64},
        b::AbstractVector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = HIGHS_DEFAULT_SOLVER,
        threads = nothing
)
    _validate_highs_tolerance(tol)

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

    A_normalized, b_normalized = normalize_lp(A_filtered, b_filtered)

    model = create_highs_model(; solver = solver, threads = threads)

    @variable(model, x[1:n])
    @variable(model, epsilon)
    @constraint(model, A_normalized * x .+ epsilon .<= b_normalized)
    # A cap above the tolerance preserves the result and keeps the LP bounded.
    @constraint(model, epsilon <= max(1.0, 2 * tol))
    @objective(model, Max, epsilon)

    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        epsilon_value = value(epsilon)
        isfinite(epsilon_value) ||
            throw(ErrorException("HiGHS returned non-finite inflation value $epsilon_value"))
        return epsilon_value > tol
    elseif status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
        # The objective is bounded, so the ambiguous status means infeasible.
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

Return `true` if the maximum slack of the `i`th inequality in `{x : Ax ≤ b}` is
at most `tol`. The slack is in input-space units.
"""
function highs_check_implicit_equality(
        A::AbstractMatrix{Float64}, b::AbstractVector{Float64}, i::Int;
        tol = HIGHS_DEFAULT_TOL,
        solver = HIGHS_DEFAULT_SOLVER,
        threads = nothing
)
    m, n = size(A)
    1 <= i <= m || throw(BoundsError("Row index $i out of bounds for matrix with $m rows"))

    scale = LinearAlgebra.norm(@view A[i, :])
    iszero(scale) && return iszero(b[i])

    A_filtered, b_filtered = filter_lp(A, b)
    A_normalized, b_normalized = normalize_lp(A_filtered, b_filtered)
    target_row = @view(A[i, :]) ./ scale
    target_rhs = b[i] / scale

    model = create_highs_model(; solver = solver, threads = threads)

    @variable(model, x[1:n])
    @variable(model, s)
    @constraints(model, begin
        A_normalized * x .<= b_normalized
        s == target_rhs - LinearAlgebra.dot(target_row, x)
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

"""Solve the common-facet LP and return its classified result and status."""
function _lp_cells_share_facet(model, left, right, hyperplane_key, mode)
    # Get the equation of the candidate hyperplane.
    n = size(left.A, 2)
    normal = Float64.(collect(hyperplane_key[1:n]))
    rhs = -Float64(hyperplane_key[n + 1])

    # Constrain the slack model to the candidate hyperplane.
    @variable(model, x[1:n])
    @variable(model, excess)
    @constraint(model, LinearAlgebra.dot(normal, x) == rhs)
    @constraint(model, excess <= 1)

    # Require positive slack for every other nonconstant inequality.
    for (A, b) in ((left.A, left.b), (right.A, right.b))
        for i in axes(A, 1)
            row = @view A[i, :]
            all(iszero, row) && continue
            _inequality_has_supporting_hyperplane(
                row, b[i], hyperplane_key) && continue
            row_norm = LinearAlgebra.norm(Float64.(row))
            normalized_row = Float64.(row) ./ row_norm
            normalized_rhs = Float64(b[i]) / row_norm
            @constraint(model,
                LinearAlgebra.dot(normalized_row, x) + mode.tol + excess <= normalized_rhs)
        end
    end

    # Positive excess over the tolerance confirms that the common face is a facet.
    @objective(model, Max, excess)
    optimize!(model)

    status = termination_status(model)
    status == MOI.OPTIMAL || return nothing, status
    return value(excess) > 0, status
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
    A_normalized, _ = normalize_lp(A_filtered, b_filtered)

    redundantIdx = []

    for i in 1:size(A_filtered, 1)
        # Two independent implicit equalities give codimension at least two.
        if highs_check_implicit_equality(
            A_filtered, b_filtered, i; tol = tol, solver = solver, threads = threads)
            push!(redundantIdx, i)
        end
        if length(redundantIdx) > 1
            # Test independence with the row rank.
            A_redundant = A_normalized[redundantIdx, :]
            if LinearAlgebra.rank(A_redundant) > 1
                return false
            end
        end
    end
    return true
end

"""
    _inequality_has_supporting_hyperplane(row, rhs, hyperplane_key; atol=1e-9)

Return `true` if the supporting hyperplane of a floating-point inequality is
the hyperplane in `hyperplane_key`.
"""
function _inequality_has_supporting_hyperplane(
        row, rhs, hyperplane_key; atol = 1.0e-9)
    # Store the affine equation `row*x - rhs = 0` as a coefficient vector.
    equation = [Float64.(row); -Float64(rhs)]

    # Normalize the equation so that scalar multiples have the same coefficients.
    pivot_index = findfirst(!iszero, equation)
    normalized = equation ./ equation[pivot_index]

    # Scale the tolerance by the larger of one and the largest coefficient magnitude.
    target_hyperplane = Float64.(collect(hyperplane_key))
    comparison_scale = max(1.0, maximum(abs, target_hyperplane))
    return maximum(abs, normalized .- target_hyperplane) <= atol * comparison_scale
end

"""
    _highs_cells_share_facet(left, right, hyperplane_key, mode)

Test whether two full-dimensional cells share a facet with the specified
supporting hyperplane.
"""
function _highs_cells_share_facet(
        left::_Cell,
        right::_Cell,
        hyperplane_key,
        mode::_HiGHS
)
    model = create_highs_model(; solver = mode.solver, threads = mode.threads)
    solve_lp = current_model -> _lp_cells_share_facet(
        current_model, left, right, hyperplane_key, mode)
    return _solve_with_glpk_fallback(solve_lp, model, "facet check")
end

"""
    _cells_share_facet(left, right, hyperplane_key, mode)

Test whether two cells share a facet supported by the hyperplane in
`hyperplane_key`. Use the backend selected by `mode`.
"""
function _cells_share_facet(left::_Cell, right::_Cell, hyperplane_key, mode)
    if mode isa _HiGHS
        return _highs_cells_share_facet(left, right, hyperplane_key, mode)
    end
    return regions_intersect_codimension_le_one(left, right; mode = mode)
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
