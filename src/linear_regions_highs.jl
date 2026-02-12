# This file contains HiGHS-based implementations for computing
# the linear regions of tropical Puiseux polynomials and rational functions.
#
# These functions provide faster alternatives to the Oscar-based implementations
# in linear_regions.jl by using HiGHS LP solver directly without creating
# polyhedron objects.

# Use AppleAccelerate on macOS for better performance
if Sys.isapple()
    try
        @eval using AppleAccelerate
    catch
        @warn "AppleAccelerate not available, HiGHS will use default BLAS"
    end
end

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
    polyhedron_highs(f::TropicalPuiseuxPoly, i::Int)

Outputs the (A, b) matrix representation of the polyhedron corresponding to
points where f is given by the linear map corresponding to the i-th monomial of f.

Returns a tuple (A, b) where the polyhedron is {x : Ax ≤ b}.

# Example
Output the polyhedron where f = max(x, y) is equal to x
```jldoctest
julia> f = TropicalPuiseuxPoly(Dict([1, 0] => 0, [0, 1] => 0), [[1, 0], [0, 1]]);

julia> A, b = polyhedron_highs(f, [1, 0])
```
"""
function polyhedron_highs(f::TropicalPuiseuxPoly, i)
    # take A to be the matrix with rows αⱼ - αᵢ for all j ≠ i, where the αᵢ are the exponents of f.
    A = mapreduce(permutedims, vcat, [Float64.(f.exp[j]) - Float64.(f.exp[i]) for j in eachindex(f)])
    # and b the vector whose j-th entry is f.coeff[αⱼ] - f.coeff[αᵢ] for all j ≠ i.
    b = [Float64(Rational(f.coeff[f.exp[i]])) - Float64(Rational(f.coeff[j])) for j in f.exp]
    # The polyhedron is then the set of points x such that Ax ≤ b.
    return (A, b)
end

@doc raw"""
    enum_linear_regions_highs(f::TropicalPuiseuxPoly; tol=HIGHS_DEFAULT_TOL)

Outputs an array of tuples ((A, b), bool) indexed by the same set as the exponents of f.
The tuple element (A, b) is the matrix representation of the linear region corresponding
to the exponent, and bool is true when this region is nonempty.

Uses HiGHS LP solver for fast feasibility checks.

# Example
Enumerates the linear regions of f = max(x, y).
```jldoctest
julia> f = TropicalPuiseuxPoly(Dict([1, 0] => 0, [0, 1] => 0), [[1, 0], [0, 1]]);

julia> enum_linear_regions_highs(f)
2-element Vector{Any}:
 ((A, b), true)
 ((A, b), true)
```
"""
function enum_linear_regions_highs(f::TropicalPuiseuxPoly; tol=HIGHS_DEFAULT_TOL)
    linear_regions = Vector()
    sizehint!(linear_regions, length(f.exp))
    for i in eachindex(f)
        A, b = polyhedron_highs(f, i)
        # Check if the polyhedron is non-empty by checking if it's feasible
        is_feasible = !highs_is_empty(A, b)
        push!(linear_regions, ((A, b), is_feasible))
    end
    return linear_regions
end

@doc raw"""
    enum_linear_regions_rat_highs(q::TropicalPuiseuxRational; tol=HIGHS_DEFAULT_TOL)

Computes the linear regions of a tropical Puiseux rational function f/g using HiGHS.

Inputs: Tropical Puiseux rational q = f/g
Output: array containing linear regions of f/g represented by (A, b) matrix pairs or arrays of pairs.

Uses HiGHS LP solver for fast intersection and full-dimensionality checks.

# Example
Enumerates the linear regions of f/g where f = max(x, y) and g = max(x+y, x+2y).
```jldoctest
julia> f = TropicalPuiseuxPoly(Dict([1, 0] => 0, [0, 1] => 0), [[1, 0], [0, 1]]);

julia> g = TropicalPuiseuxPoly(Dict([1, 1] => 0, [1, 2] => 0), [[1, 1], [1, 2]]);

julia> q = TropicalPuiseuxRational(f, g);

julia> enum_linear_regions_rat_highs(q)
4-element Vector{Any}:
 (A1, b1)
 (A2, b2)
 (A3, b3)
 (A4, b4)
```
"""
function enum_linear_regions_rat_highs(q::TropicalPuiseuxRational; tol=HIGHS_DEFAULT_TOL)
    f = q.num
    g = q.den
    # first, compute the linear regions of f and g.
    lin_f = enum_linear_regions_highs(f; tol=tol)
    lin_g = enum_linear_regions_highs(g; tol=tol)

    # next, check which for repetitions of the linear map corresponding to f/g on intersections of the linear regions computed above.
    function check_linear_repetitions()
        # Key insight: Use linear map [c, α] tuple as dictionary KEY (what we're tracking)
        # and store list of (A, b) regions as VALUE
        # This avoids floating-point comparison issues with (A, b) tuples
        map_to_regions = Dict()  # (c, α) -> [(A, b), ...]

        # We need to check for pairwise intersection of each polytope, by iterating over
        for i in eachindex(f)
            for j in eachindex(g)
                # we only need to do the checks on linear regions that are attained by f and g
                if lin_f[i][2] && lin_g[j][2]
                    # Get the matrix representations
                    A1, b1 = lin_f[i][1]
                    A2, b2 = lin_g[j][1]

                    # check if the polytopes intersect on a full-dimensional region
                    if highs_intersect_is_full_dimensional(A1, b1, A2, b2; tol=tol)
                        # Compute the intersection by concatenating constraints
                        A_intersect = vcat(A1, A2)
                        b_intersect = vcat(b1, b2)
                        Ab = (A_intersect, b_intersect)

                        # Compute the linear map for this region
                        c = Rational(f.coeff[f.exp[i]]) - Rational(g.coeff[g.exp[j]])
                        α = f.exp[i] - g.exp[j]
                        linear_map_key = (c, α)

                        # Group regions by their linear map
                        if haskey(map_to_regions, linear_map_key)
                            push!(map_to_regions[linear_map_key], Ab)
                        else
                            map_to_regions[linear_map_key] = [Ab]
                        end
                    end
                end
            end
        end

        # Check for repetitions: if any linear map has multiple regions
        exists_reps = any(length(regions) > 1 for regions in values(map_to_regions))

        if !exists_reps
            # Return all regions (flatten the values)
            all_regions = vcat(values(map_to_regions)...)
            return map_to_regions, all_regions, false
        else
            # Return only the groups that have repetitions
            reps = [(lmap, regions) for (lmap, regions) in map_to_regions]
            return map_to_regions, reps, true
        end
    end

    map_to_regions, reps, exists_reps = check_linear_repetitions()

    # if there are no repetitions, then the linear regions are just the non-empty intersections of linear regions of f and linear regions of g
    if !exists_reps
        lin_regions = reps  # reps contains all_regions when there are no repetitions
    # if there are repetitions then we will need to find connected components of the union of the polytopes on which repetitions occur.
    else
        # Initialise the array lin_regions. This will contain the true linear regions of f/g
        lin_regions = []
        # first find all pairwise intersections of polytopes.
        for (lmap, regions) in reps
            # if regions has length 1 then there is no other linear region with the same linear map
            if length(regions) == 1
                append!(lin_regions, regions)
            else
                # otherwise, we check for intersections in the set of linear regions with a given map
                has_intersect = Dict()
                # iterate over unordered pairs of (distinct) elements of regions
                for (Ab1, Ab2) in Combinatorics.combinations(regions, 2)
                    A1, b1 = Ab1
                    A2, b2 = Ab2

                    # Check if the two polyhedra intersect (feasibly)
                    A_check = vcat(A1, A2)
                    b_check = vcat(b1, b2)
                    intersects = !highs_is_empty(A_check, b_check)

                    # add true to the dictionary if the intersection is nonempty and false otherwise
                    has_intersect[(Ab1, Ab2)] = intersects
                end
                # now find transitive closure of the relation given by dictionary has_intersect
                # and append the corresponding components to lin_regions
                append!(lin_regions, components(regions, has_intersect))
            end
        end
    end
    return lin_regions
end
