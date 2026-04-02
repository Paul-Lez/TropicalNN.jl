# This file contains HiGHS-based implementations for computing
# the linear regions of tropical Puiseux polynomials and rational functions.
#
# These functions provide faster alternatives to the Oscar-based implementations
# in linear_regions.jl by using HiGHS LP solver directly without creating
# polyhedron objects.

using HiGHS
using JuMP

# Tolerance for numerical comparisons
const HIGHS_DEFAULT_TOL = 1e-6

# Helper function to create optimized HiGHS model
function create_highs_model()
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    # Use interior point solver ("hipo") for better performance on Mac
    set_attribute(model, "solver", "hipo")
    return model
end

"""
    highs_is_empty(A::Matrix{Float64}, b::Vector{Float64})

Check if polyhedron {x : Ax ≤ b} is empty via LP feasibility using HiGHS.

Solves: find x such that Ax ≤ b
If infeasible, polyhedron is empty.
"""
function highs_is_empty(A::Matrix{Float64}, b::Vector{Float64})
    m, n = size(A)

    model = create_highs_model()

    @variable(model, x[1:n])
    @constraint(model, A * x .<= b)

    optimize!(model)

    status = termination_status(model)
    return status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
end

"""
    highs_is_full_dimensional(A::Matrix{Float64}, b::Vector{Float64}; tol=HIGHS_DEFAULT_TOL)

Check if polyhedron {x : Ax ≤ b} is full dimensional via LP using HiGHS.

Solves: max ε such that Ax + ε·1 ≤ b
If optimal ε > tol, polyhedron has nonempty interior (is full dimensional).

Note: Filters out trivial constraints (all-zero rows) before checking, as these
don't restrict the polyhedron but prevent the inflation test from working.
"""
function highs_is_full_dimensional(A::Matrix{Float64}, b::Vector{Float64}; tol=HIGHS_DEFAULT_TOL)
    m, n = size(A)

    # Filter out trivial constraints (rows that are all zeros)
    # These are constraints like 0*x ≤ b which don't actually restrict the space
    non_trivial_rows = [i for i in 1:m if !all(abs.(A[i, :]) .< tol)]

    # If all constraints are trivial, the polyhedron is all of R^n (full dimensional)
    if isempty(non_trivial_rows)
        return true
    end

    A_filtered = A[non_trivial_rows, :]
    b_filtered = b[non_trivial_rows]

    model = create_highs_model()

    @variable(model, x[1:n])
    @variable(model, ε)
    @constraint(model, A_filtered * x .+ ε .<= b_filtered)
    @objective(model, Max, ε)

    optimize!(model)

    status = termination_status(model)
    if status == MOI.OPTIMAL
        return value(ε) > tol
    elseif status == MOI.DUAL_INFEASIBLE
        # Dual infeasible means primal is unbounded
        # If ε can be arbitrarily large, the polyhedron is definitely full dimensional
        return true
    else
        # If infeasible, polyhedron is not full dimensional (or is empty)
        return false
    end
end

"""
    highs_intersect_is_full_dimensional(A1::Matrix{Float64}, b1::Vector{Float64},
                                        A2::Matrix{Float64}, b2::Vector{Float64};
                                        tol=HIGHS_DEFAULT_TOL)

Check if intersection of two polyhedra is full dimensional via LP using HiGHS.

Concatenates constraints and checks if the combined polyhedron is full dimensional.
"""
function highs_intersect_is_full_dimensional(A1::Matrix{Float64}, b1::Vector{Float64},
                                             A2::Matrix{Float64}, b2::Vector{Float64};
                                             tol=HIGHS_DEFAULT_TOL)
    @assert size(A1, 2) == size(A2, 2) "Ambient dimensions must match"

    # Compute intersection by concatenating constraints
    A_combined = vcat(A1, A2)
    b_combined = vcat(b1, b2)

    # Check if the intersection is full dimensional
    return highs_is_full_dimensional(A_combined, b_combined; tol=tol)
end

@doc raw"""
    polyhedron_highs(f::Signomial, i::Int)

Outputs the (A, b) matrix representation of the polyhedron corresponding to
points where f is given by the linear map corresponding to the i-th monomial of f.

Returns a tuple (A, b) where the polyhedron is {x : Ax ≤ b}.

# Example
Output the polyhedron where f = max(x, y) is equal to x
```jldoctest
julia> f = Signomial(Dict([1, 0] => 0, [0, 1] => 0), [[1, 0], [0, 1]]);

julia> A, b = polyhedron_highs(f, 1);

julia> size(A)
(1, 2)
```
"""
function polyhedron_highs(f::AbstractSignomial, i)
    exp_i   = get_exp(f, i)
    coeff_i = get_coeff(f, i)
    # take A to be the matrix with rows αⱼ - αᵢ for all j ≠ i, where the αᵢ are the exponents of f.
    rows = [Vector{Float64}(get_exp(f, j)) - Vector{Float64}(exp_i) for j in Base.eachindex(f) if j != i]
    # Single-monomial polynomial: the whole R^n is the unique region.
    if isempty(rows)
        n = nvars(f)
        return (Matrix{Float64}(undef, 0, n), Float64[])
    end
    A = Matrix{Float64}(mapreduce(permutedims, vcat, rows))
    # and b the vector whose j-th entry is f.coeff[αᵢ] - f.coeff[αⱼ] for all j ≠ i.
    b = [Float64(Rational(coeff_i)) - Float64(Rational(get_coeff(f, j))) for j in Base.eachindex(f) if j != i]
    # The polyhedron is then the set of points x such that Ax ≤ b.
    return (A, b)
end

@doc raw"""
    enum_linear_regions_highs(f::Signomial; tol=HIGHS_DEFAULT_TOL)

Outputs an array of tuples ((A, b), bool) indexed by the same set as the exponents of f.
The tuple element (A, b) is the matrix representation of the linear region corresponding
to the exponent, and bool is true when this region is nonempty.

Uses HiGHS LP solver for fast feasibility checks.

# Example
Enumerates the linear regions of f = max(x, y).
```jldoctest
julia> f = Signomial(Dict([1, 0] => 0, [0, 1] => 0), [[1, 0], [0, 1]]);

julia> length(enum_linear_regions_highs(f))
2
```
"""
function enum_linear_regions_highs(f::AbstractSignomial; tol=HIGHS_DEFAULT_TOL)
    linear_regions = Vector{Tuple{Tuple{Matrix{Float64},Vector{Float64}},Bool}}()
    sizehint!(linear_regions, length(f))
    for i in Base.eachindex(f)
        A, b = polyhedron_highs(f, i)
        # Check if the polyhedron is non-empty by checking if it's feasible
        is_feasible = !highs_is_empty(A, b)
        push!(linear_regions, ((A, b), is_feasible))
    end
    return linear_regions
end

@doc raw"""
    enum_linear_regions_rat_highs(q::RationalSignomial; tol=HIGHS_DEFAULT_TOL)

Computes the linear regions of a tropical Puiseux rational function f/g using HiGHS.

Faster alternative to `enum_linear_regions_rat`: uses the HiGHS LP solver directly
instead of building Oscar `Polyhedron` objects. The return type mirrors that of
`enum_linear_regions_rat` — a `LinearRegions` object — so both backends can be used
interchangeably. Each `LinearRegion` stores one or more `(A, b)` matrix pairs (instead
of `Oscar.Polyhedron` objects) representing the convex pieces of that region.

# Arguments
- `q::RationalSignomial`: The rational function whose linear regions are computed.
- `tol`: Numerical tolerance for LP feasibility and full-dimensionality checks.

# Returns
A `LinearRegions` object. Each element is a `LinearRegion` whose `regions` field holds
the `(A, b)` pairs (where the polyhedron is `{x : Ax ≤ b}`) making up that region.
"""
function enum_linear_regions_rat_highs(q::RationalSignomial; tol=HIGHS_DEFAULT_TOL)
    f = q.num
    g = q.den
    # first, compute the linear regions of f and g.
    lin_f = enum_linear_regions_highs(f; tol=tol)
    lin_g = enum_linear_regions_highs(g; tol=tol)

    # Group all full-dimensional intersections by the linear map they realise.
    # map_to_regions: (c, α) -> Vector of (A, b) pairs
    map_to_regions = Dict()

    for i in Base.eachindex(f)
        for j in Base.eachindex(g)
            if lin_f[i][2] && lin_g[j][2]
                A1, b1 = lin_f[i][1]
                A2, b2 = lin_g[j][1]

                if highs_intersect_is_full_dimensional(A1, b1, A2, b2; tol=tol)
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
                has_intersect[(Ab1, Ab2)] = !highs_is_empty(A_check, b_check)
            end
            for component in components(regions, has_intersect)
                push!(lr_list, LinearRegion(convert(Vector{Ab_type}, component)))
            end
        end
    end
    return LinearRegions(lr_list)
end
