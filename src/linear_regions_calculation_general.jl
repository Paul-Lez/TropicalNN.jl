# Shared implementation for computing linear regions of tropical Puiseux
# polynomials and rational functions.

const HIGHS_DEFAULT_TOL = 1e-6

abstract type LinearRegionsCalculationMode end

struct _Oscar <: LinearRegionsCalculationMode end

Base.@kwdef struct _HiGHS <: LinearRegionsCalculationMode
    tol::Float64 = HIGHS_DEFAULT_TOL
    solver::String = "hipo"
end

struct _Polyhedra
    A::Matrix{Float64}
    b::Vector{Float64}
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

function get_matrix(region::Oscar.Polyhedron; mode::_Oscar)
    return Float64.(Oscar.halfspace_matrix_pair(Oscar.facets(region)).A)
end

function get_matrix(region::_Polyhedra; mode::_HiGHS)
    return region.A
end

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
    polyhedron(f::Signomial, i::Int)

Outputs the polyhedron corresponding to points where f is given by the
linear map corresponding to the i-th monomial of f.
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

Compute all linear regions for signomial `f`.

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
alorithm indicated by `mode`.
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
