# This file contains HiGHS-based implementations for computing
# the linear regions of tropical Puiseux polynomials and rational functions.
#
# These functions provide faster alternatives to the Oscar-based implementations
# in linear_regions.jl by using HiGHS LP solver directly without creating
# polyhedron objects.

using HiGHS
using JuMP

# Helper function to create optimized HiGHS model
function create_highs_model(; solver = "hipo")
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    # Use interior point solver ("hipo") for better performance on Mac
    set_attribute(model, "solver", solver)
    return model
end

"""
    highs_is_empty(A::Matrix{Float64}, b::Vector{Float64}; solver="hipo")

Check if polyhedron {x : Ax ≤ b} is empty via LP feasibility using HiGHS.

Solves: find x such that Ax ≤ b
If infeasible, polyhedron is empty.
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

Check if polyhedron {x : Ax ≤ b} is full dimensional via LP using HiGHS.

Solves: max ε such that Ax + ε·1 ≤ b
If optimal ε > tol, polyhedron has nonempty interior (is full dimensional).

Note: Filters out trivial constraints (all-zero rows) before checking, as these
don't restrict the polyhedron but prevent the inflation test from working.
"""
function highs_is_full_dimensional(
        A::Matrix{Float64},
        b::Vector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = "hipo"
)

    m, n = size(A)

    # Filter out exactly trivial constraints. Rows with small nonzero entries
    # still define real halfspaces and should not be dropped based on `tol`.
    zero_rows = [i for i in 1:m if all(iszero, A[i, :])]
    any(i -> b[i] < 0, zero_rows) && return false
    non_trivial_rows = [i for i in 1:m if !(i in zero_rows)]

    # If all constraints are trivial, the polyhedron is all of R^n (full dimensional)
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
        # Dual infeasible means primal is unbounded
        # If ε can be arbitrarily large, the polyhedron is definitely full dimensional
        return true
    elseif status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
        # If infeasible, polyhedron is not full dimensional (or is empty)
        return false
    end
    throw(ErrorException("HiGHS full-dimensionality check ended with unexpected status $status"))
end

"""
    highs_intersect_is_full_dimensional(A1::Matrix{Float64}, b1::Vector{Float64},
                                        A2::Matrix{Float64}, b2::Vector{Float64};
                                        tol=HIGHS_DEFAULT_TOL, solver="hipo")

Check if intersection of two polyhedra is full dimensional via LP using HiGHS.

Concatenates constraints and checks if the combined polyhedron is full dimensional.
"""
function highs_intersect_is_full_dimensional(A1::Matrix{Float64}, b1::Vector{Float64},
        A2::Matrix{Float64}, b2::Vector{Float64};
        tol = HIGHS_DEFAULT_TOL,
        solver = "hipo")
    size(A1, 2) == size(A2, 2) ||
        throw(DimensionMismatch("Ambient dimensions must match, got $(size(A1, 2)) and $(size(A2, 2))"))

    # Compute intersection by concatenating constraints
    A_combined = vcat(A1, A2)
    b_combined = vcat(b1, b2)

    # Check if the intersection is full dimensional
    return highs_is_full_dimensional(A_combined, b_combined; tol = tol, solver = solver)
end

@doc raw"""
    polyhedron_highs(f::Signomial, i::Int)

Return `(A, b)` for the region where the `i`th monomial of `f` dominates,
encoded as `{x : Ax <= b}`.
"""
function polyhedron_highs(f::AbstractSignomial, i)
    A, b = _linear_region_constraints(f, i, Float64; empty_rows = 0)
    # The polyhedron is then the set of points x such that Ax ≤ b.
    return (A, b)
end

@doc raw"""
    enum_linear_regions_highs(f::Signomial; tol=HIGHS_DEFAULT_TOL, solver="hipo")

Return `((A, b), is_feasible)` for each monomial region of `f`, using HiGHS
for feasibility checks. `(A, b)` encodes `{x : Ax <= b}`.
"""
function enum_linear_regions_highs(
        f::AbstractSignomial;
        tol = HIGHS_DEFAULT_TOL,
        solver = "hipo"
)
    linear_regions = Vector{Tuple{Tuple{Matrix{Float64}, Vector{Float64}}, Bool}}()
    sizehint!(linear_regions, length(f))
    for i in Base.eachindex(f)
        A, b = polyhedron_highs(f, i)
        # Check if the polyhedron is non-empty by checking if it's feasible
        is_feasible = !highs_is_empty(A, b; solver = solver)
        push!(linear_regions, ((A, b), is_feasible))
    end
    return linear_regions
end

@doc raw"""
    enum_linear_regions_rat_highs(q::RationalSignomial; tol=HIGHS_DEFAULT_TOL, solver="hipo")

Compute the linear regions of `q` using HiGHS LP checks.

Returns a `LinearRegions` object whose entries store `(A, b)` pairs for convex
pieces `{x : Ax <= b}`. `tol` is used for feasibility and full-dimensionality
checks.
"""
function enum_linear_regions_rat_highs(
        q::RationalSignomial;
        tol = HIGHS_DEFAULT_TOL,
        solver = "hipo"
)
    f = q.num
    g = q.den
    # first, compute the linear regions of f and g.
    lin_f = enum_linear_regions_highs(f; tol = tol, solver = solver)
    lin_g = enum_linear_regions_highs(g; tol = tol, solver = solver)

    # Group all full-dimensional intersections by the linear map they realise.
    # map_to_regions: (c, α) -> Vector of (A, b) pairs
    map_to_regions = Dict()

    for i in Base.eachindex(f)
        for j in Base.eachindex(g)
            if lin_f[i][2] && lin_g[j][2]
                A1, b1 = lin_f[i][1]
                A2, b2 = lin_g[j][1]

                if highs_intersect_is_full_dimensional(
                    A1, b1, A2, b2; tol = tol, solver = solver)
                    Ab = (vcat(A1, A2), vcat(b1, b2))
                    c = Rational(get_coeff(f, i)) - Rational(get_coeff(g, j))
                    α = collect(get_exp(f, i)) - collect(get_exp(g, j))
                    key = (c, α)
                    if haskey(map_to_regions, key)
                        push!(map_to_regions[key], Ab)
                    else
                        map_to_regions[key] = [Ab]
                    end
                end
            end
        end
    end

    # Build a LinearRegions result, grouping (A, b) pairs into connected components.
    # Each connected component becomes one LinearRegion.  The structure mirrors the
    # Oscar-based enum_linear_regions_rat so that callers can treat both backends uniformly.
    Ab_type = Tuple{Matrix{Float64}, Vector{Float64}}
    lr_list = LinearRegion{Ab_type}[]
    for (_, regions) in map_to_regions
        if length(regions) == 1
            push!(lr_list, LinearRegion(regions))
        else
            # Check pairwise feasibility of intersections to find connected components
            has_intersect = Dict()
            for (Ab1, Ab2) in Combinatorics.combinations(regions, 2)
                A_check = vcat(Ab1[1], Ab2[1])
                b_check = vcat(Ab1[2], Ab2[2])
                has_intersect[(Ab1, Ab2)] = !highs_is_empty(A_check, b_check; solver = solver)
            end
            for component in components(regions, has_intersect)
                push!(lr_list, LinearRegion(convert(Vector{Ab_type}, component)))
            end
        end
    end
    return LinearRegions(lr_list)
end
