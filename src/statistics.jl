# Geometry helpers.

"""
    intersect_reps(rep_1, rep_2) -> [A, b]

Stack two `[A, b]` halfspace representations to represent their intersection.
"""
function intersect_reps(rep_1, rep_2)
    return [vcat(rep_1[1], rep_2[1]), vcat(rep_1[2], rep_2[2])]
end

"""
    _hrep_is_full_dimensional(A, b, mode)

Check whether the polyhedron `Ax ≤ b` is full-dimensional.
"""
function _hrep_is_full_dimensional(A, b, mode::LinearRegionsCalculationMode)
    return is_full_dimensional(make_polyhedron(A, b; mode = mode); mode = mode)
end

"""
    m_reps(f::Signomial; mode::LinearRegionsCalculationMode=OscarMode()) -> Dict

Return `[A, b]` for each full-dimensional monomial region of `f`.
The `"m_reps"` and `"f_indices"` entries contain the representations and
matching monomial indices.
"""
function m_reps(
        f::Signomial;
        mode::LinearRegionsCalculationMode = OscarMode()
)
    reps = Dict("m_reps" => [], "f_indices" => [])
    for i in Base.eachindex(f)
        A, b = _linear_region_constraints(
            f,
            i,
            OSCAR_POLYHEDRON_COEFF_TYPE;
            include_self = true
        )
        if _hrep_is_full_dimensional(A, b, mode)
            push!(reps["m_reps"], [A, b])
            push!(reps["f_indices"], i)
        end
    end
    return reps
end

"""
    m_reps(f::RationalSignomial; mode::LinearRegionsCalculationMode=OscarMode()) -> Dict

Return `[A, b]` for each full-dimensional numerator-denominator region
intersection of `f`. The `"f_indices"` entry contains the matching index pairs.
"""
function m_reps(
        f::RationalSignomial;
        mode::LinearRegionsCalculationMode = OscarMode()
)
    n_reps = m_reps(f.num; mode = mode)
    d_reps = m_reps(f.den; mode = mode)
    reps = Dict("m_reps" => [], "f_indices" => [])
    for (n_m_rep, n_f_idx) in zip(n_reps["m_reps"], n_reps["f_indices"])
        for (d_m_rep, d_f_idx) in zip(d_reps["m_reps"], d_reps["f_indices"])
            int_rep = intersect_reps(n_m_rep, d_m_rep)
            if _hrep_is_full_dimensional(int_rep[1], int_rep[2], mode)
                push!(reps["m_reps"], [int_rep[1], int_rep[2]])
                push!(reps["f_indices"], [n_f_idx, d_f_idx])
            end
        end
    end
    return reps
end

"""
    polyhedra_from_reps(reps::Dict, oscar::Bool=false) -> Vector

Convert the output of [`m_reps`](@ref) to exact Oscar polyhedra when
`oscar=true` or exact CDDLib-backed polyhedra otherwise.
"""
function polyhedra_from_reps(reps, oscar::Bool = false)
    if oscar
        return [Oscar.polyhedron(m_rep[1], m_rep[2]) for m_rep in reps["m_reps"]]
    else
        return [Polyhedra.polyhedron(Polyhedra.hrep(m_rep[1], m_rep[2]), CDDLib.Library(:exact))
                for m_rep in reps["m_reps"]]
    end
end

"""
    get_linear_maps(f::Union{Signomial,RationalSignomial}, f_indices) -> Vector

Return `[constant, exponent_vector]` for each region index in `f_indices`.
"""
function get_linear_maps(f::Union{Signomial, RationalSignomial}, f_indices)
    linear_maps = Vector{Vector{Any}}()
    for f_idx in f_indices
        if length(f_idx) == 1
            push!(linear_maps, [Rational(get_coeff(f, f_idx)), collect(get_exp(f, f_idx))])
        else
            numr, denr = f.num, f.den
            push!(linear_maps,
                [Rational(get_coeff(numr, f_idx[1])) - Rational(get_coeff(denr, f_idx[2])),
                    collect(get_exp(numr, f_idx[1])) - collect(get_exp(denr, f_idx[2]))])
        end
    end
    return linear_maps
end

"""
    get_linear_regions(polyhedra, linear_maps) -> Dict

Group `polyhedra` by affine formula. Each result entry stores the matching
polyhedra, which can have a disconnected union.
"""
function get_linear_regions(polyhedra, linear_maps)
    linear_regions = Dict()
    for (linear_map, poly) in zip(linear_maps, polyhedra)
        if !haskey(linear_regions, linear_map)
            linear_regions[linear_map] = Dict("polyhedra" => [poly])
        else
            push!(linear_regions[linear_map]["polyhedra"], poly)
        end
    end
    return linear_regions
end

# Region grouping.

@doc raw"""
    separate_components(linear_regions::Dict)

Split each affine-map entry into components connected through codimension-zero
or codimension-one intersections.
"""
function separate_components(linear_regions::Dict)
    region_components = Dict()
    for (linear_map, value) in linear_regions
        polys = value["polyhedra"]
        # Do not join pieces through higher-codimension contacts.
        has_intersect = Dict()
        for (poly1, poly2) in Combinatorics.combinations(polys, 2)
            intersection = Oscar.intersect(poly1, poly2)
            has_intersect[(poly1, poly2)] = Oscar.is_feasible(intersection) &&
                                            Oscar.codim(intersection) <= 1
        end
        region_components[linear_map] = components(polys, has_intersect)
    end
    return region_components
end

@doc raw"""
    map_statistic(statistic, f; mode::LinearRegionsCalculationMode=OscarMode())

Apply a function `statistic` to the linear-region dictionary for `f`.
"""
function map_statistic(
        statistic::Function,
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    # Get the unbounded halfspace representations.
    reps = m_reps(f; mode = mode)
    # Construct exact Oscar polyhedra.
    polys = polyhedra_from_reps(reps, true)
    # Get the affine formula for each polyhedron.
    linear_maps = get_linear_maps(f, reps["f_indices"])
    # Split sets that do not have codimension-zero or codimension-one links.
    linear_regions = separate_components(get_linear_regions(polys, linear_maps))
    return statistic(linear_regions)
end

# Region measures.

@doc raw"""
    interior_points(polys::Array)

Return the finite-vertex centroid of each polyhedron. Return `nothing` when a
polyhedron has no vertices.
"""
function interior_points(polys::Array)
    component_interiors = Vector{Any}()
    for poly in polys
        # Average the finite vertices.
        vertices = Oscar.vertices(poly)
        if isempty(vertices)
            push!(component_interiors, nothing)
        else
            push!(component_interiors, sum(vertices) / length(vertices))
        end
    end
    return component_interiors
end

@doc raw"""
    interior_points(linear_regions::Dict)

Return finite-vertex centroids for the polyhedra in `linear_regions`.
"""
function interior_points(linear_regions::Dict)
    result = Dict()
    for (linear_map, components) in linear_regions
        components_interiors = Vector{Any}()
        for polys in components
            push!(components_interiors, interior_points(polys))
        end
        result[linear_map] = components_interiors
    end
    return result
end

@doc raw"""
    interior_points(f::Union{Signomial,RationalSignomial};
                    mode::LinearRegionsCalculationMode=OscarMode())

Return finite-vertex centroids for the linear-region polyhedra of `f`.
"""
function interior_points(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return map_statistic(interior_points, f; mode = mode)
end

@doc raw"""
    bounds(polys::Array)

Check boundedness for each polyhedron in `polys`.
"""
function bounds(polys::Array)
    return Oscar.is_bounded.(polys)
end

@doc raw"""
    bounds(linear_regions::Dict)

Check boundedness for all polyhedra in `linear_regions`.
"""
function bounds(linear_regions::Dict)
    bounded = Dict()
    for (linear_map, components) in linear_regions
        components_bounded = Vector{Vector{Bool}}()
        for polys in components
            push!(components_bounded, bounds(polys))
        end
        bounded[linear_map] = components_bounded
    end
    return bounded
end

@doc raw"""
    bounds(f::Union{Signomial,RationalSignomial};
           mode::LinearRegionsCalculationMode=OscarMode())

Check boundedness for the linear regions of `f`.
"""
function bounds(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return map_statistic(bounds, f; mode = mode)
end

@doc raw"""
    volumes(polys::Array)

Return each polyhedron's `Float64` volume. Return `Inf` for an unbounded
polyhedron.
"""
function volumes(polys::Array)
    bds = bounds(polys)
    vols = Vector{Float64}()
    for (poly, bd) in zip(polys, bds)
        if bd
            push!(vols, Float64.(Oscar.volume(poly)))
        else
            push!(vols, Inf)
        end
    end
    return vols
end

@doc raw"""
    volumes(linear_regions::Dict)

Return one volume for each linear region by summing the volumes
of the (full-dimensional) polyhedra that it the union of.
"""
function volumes(linear_regions::Dict)
    vols = Dict()
    for (linear_map, components) in linear_regions
        components_vols = Vector{Float64}()
        for polys in components
            push!(components_vols, sum(volumes(polys)))
        end
        vols[linear_map] = components_vols
    end
    return vols
end

@doc raw"""
    volumes(f::Union{Signomial,RationalSignomial};
            mode::LinearRegionsCalculationMode=OscarMode())

Return the volumes of the linear regions of `f`.
"""
function volumes(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return map_statistic(volumes, f; mode = mode)
end

@doc raw"""
    polyhedron_counts(linear_regions::Dict)

Return the number of convex pieces in each linear region.
"""
function polyhedron_counts(linear_regions::Dict)
    poly_counts = Dict(linear_map => [length(component) for component in components]
    for (linear_map, components) in linear_regions)
    return poly_counts
end

@doc raw"""
    polyhedron_counts(f::Union{Signomial,RationalSignomial};
                      mode::LinearRegionsCalculationMode=OscarMode())

Return the number of convex pieces in each linear region of
`f`.
"""
function polyhedron_counts(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return map_statistic(polyhedron_counts, f; mode = mode)
end

@doc raw"""
    get_graph(linear_regions::Dict)

Construct the facet-adjacency graph of `linear_regions`, with one vertex per linear region.
Edges correspond to linear regions intersecting along a codimension one edge.
"""
function get_graph(linear_regions::Dict)
    function connection(polys_1, polys_2)
        for poly_1 in polys_1
            ndim = Oscar.ambient_dim(poly_1)
            for poly_2 in polys_2
                intersection = Oscar.intersect(poly_1, poly_2)
                # Keep facet intersections.
                if Oscar.dim(intersection) == ndim - 1
                    return intersection
                end
            end
        end
        return nothing
    end
    # Flatten the component dictionary.
    region_polys = Vector{Any}()
    linear_maps = Vector{Any}()
    for (linear_map, components) in linear_regions
        for component in components
            push!(region_polys, component)
            push!(linear_maps, linear_map)
        end
    end

    g = MetaGraph(Graphs.Graph(); label_type = Int, vertex_data_type = Dict,
        edge_data_type = Dict, graph_data = nothing)

    num_regions = length(region_polys)

    for k in 1:num_regions
        # Store one vertex centroid and volume for each convex piece.
        g[k] = Dict("interior_point" => interior_points(region_polys[k]), "volume" =>
            volumes(region_polys[k]))
    end
    # Join regions that share a facet.
    for k in 1:(num_regions - 1)
        for l in (k + 1):num_regions
            int = connection(region_polys[k], region_polys[l])
            if int != nothing
                g[k, l] = Dict("linear maps" => [linear_maps[k], linear_maps[l]], "intersection" =>
                    int)
            end
        end
    end
    return g
end

"""
    _edge_direction(edge_intersection)

Return a direction vector for the intersection of two-dimensional polyhedra.
"""
function _edge_direction(edge_intersection)
    vertices = Oscar.vertices(edge_intersection)
    # TODO: perhaps here we should throw an error if the intersection is not one dimensional (here
    # we're implicitely assuming this and always use it in that context).
    if length(vertices) >= 2
        return [vertices[2][1] - vertices[1][1], vertices[2][2] - vertices[1][2]]
    end

    rays = Oscar.rays(edge_intersection)
    if !isempty(rays)
        return [rays[1][1], rays[1][2]]
    end

    return nothing
end

"""
    _edge_gradient(direction)

Return the slope of a vector `direction`, using `Inf` for a vertical direction.
"""
function _edge_gradient(direction)
    direction === nothing && return nothing
    iszero(direction[1]) && return Inf
    return direction[2] / direction[1]
end

"""
    _normalized_direction(direction)

Normalize `direction` as a `Float64` vector.
"""
function _normalized_direction(direction)
    # TODO: Maybe here we should just throw an error?
    direction === nothing && return nothing
    d = Float64.(direction)
    norm_d = sqrt(sum(x -> x^2, d))
    iszero(norm_d) && return nothing
    return d ./ norm_d
end

"""
    _canonical_direction(direction)

Normalize `direction` and make its first nonzero component positive.
(note that direction is a vector representing the direction of an edge, so we can replace it by its negative).
"""
function _canonical_direction(direction)
    d = _normalized_direction(direction)
    # TODO: Maybe here we should just throw an error?
    d === nothing && return nothing
    for x in d
        if !iszero(x)
            return x < 0 ? -d : d
        end
    end
    return d
end

@doc raw"""
    get_graph(f::Union{Signomial,RationalSignomial};
              mode::LinearRegionsCalculationMode=OscarMode())

Return the facet-adjacency graph for the linear regions of `f`.
"""
function get_graph(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return map_statistic(get_graph, f; mode = mode)
end

@doc raw"""
    edge_count(f::Union{Signomial,RationalSignomial};
               mode::LinearRegionsCalculationMode=OscarMode())

Return the number of adjacencies between the two-dimensional linear regions
of `f`.
This only supports bivariate expressions `f`.
"""
function edge_count(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    if nvars(f) != 2
        throw(
            ArgumentError(
            "edge_count is only supported for bivariate (2-variable) tropical polynomials or rational maps.",
        ),
        )
    end
    return Graphs.ne(get_graph(f; mode = mode))
end

"""
    edge_gradients(g::MetaGraph)

Return the slopes of the boundaries between polyhedra. The `"full"` entry contains all
slopes; the other entries group slopes by finite boundary vertex.
"""
function _edge_gradients(g::MetaGraph)
    function add_g(gs, vertex, grad)
        if haskey(gs, vertex)
            push!(gs[vertex], grad)
        else
            gs[vertex] = Any[grad]
        end
        return gs
    end
    gs_full = Vector{Any}()
    gs_with_source = Dict()
    for e in collect(edge_labels(g))
        e_int = g[e[1], e[2]]["intersection"]
        vs = Oscar.vertices(e_int)
        grad = _edge_gradient(_edge_direction(e_int))
        grad === nothing && continue
        if length(vs) == 1
            gs_with_source = add_g(gs_with_source, vs[1], grad)
        elseif length(vs) >= 2
            gs_with_source = add_g(gs_with_source, vs[1], grad)
            gs_with_source = add_g(gs_with_source, vs[2], grad)
        end
        push!(gs_full, grad)
    end
    gs_with_source["full"] = gs_full
    return gs_with_source
end

@doc raw"""
    edge_gradients(f::Union{Signomial,RationalSignomial};
                   mode::LinearRegionsCalculationMode=OscarMode())

Return the slopes of the boundaries between two-dimensional linear regions
of `f`.
"""
function edge_gradients(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    if nvars(f) != 2
        throw(
            ArgumentError(
            "edge_gradients is only supported for bivariate (2-variable) tropical polynomials or rational maps.",
        ),
        )
    end
    return _edge_gradients(get_graph(f; mode = mode))
end

"""
    _edge_lengths(g::MetaGraph)

Return finite two-dimensional boundary lengths. The `"full"` entry contains
all lengths; the other entries group lengths by boundary vertex.
"""
function _edge_lengths(g::MetaGraph)
    function add_l(ls, vertex, len)
        if haskey(ls, vertex)
            push!(ls[vertex], len)
        else
            ls[vertex] = [len]
        end
        return ls
    end
    ls_full = Vector{Float64}()
    ls_with_source = Dict()
    for e in collect(edge_labels(g))
        e_int = g[e[1], e[2]]["intersection"]
        vs = Oscar.vertices(e_int)
        if length(vs) >= 2
            len = sqrt(Float64((vs[1][1] - vs[2][1])^2 + (vs[1][2] - vs[2][2])^2))
            ls_with_source = add_l(ls_with_source, vs[1], len)
            ls_with_source = add_l(ls_with_source, vs[2], len)
            push!(ls_full, len)
        end
    end
    ls_with_source["full"] = ls_full
    return ls_with_source
end

@doc raw"""
    edge_lengths(f::Union{Signomial,RationalSignomial};
                 mode::LinearRegionsCalculationMode=OscarMode())

Return the finite boundary lengths between two-dimensional linear regions of
`f`.
"""
function edge_lengths(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    if nvars(f) != 2
        throw(
            ArgumentError(
            "edge_lengths is only supported for bivariate (2-variable) tropical polynomials or rational maps.",
        ),
        )
    end
    return _edge_lengths(get_graph(f; mode = mode))
end

"""
    _edge_directions(g::MetaGraph)

Return normalized boundary directions. The `"full"` entry
contains canonical directions; the other entries group outgoing directions
by boundary vertex.
"""
function _edge_directions(g::MetaGraph)
    function add_d(ds, vertex, d)
        if haskey(ds, vertex)
            push!(ds[vertex], d)
        else
            ds[vertex] = [d]
        end
        return ds
    end
    ds_full = Vector{Vector{Float64}}()
    ds_with_source = Dict()
    for e in collect(edge_labels(g))
        e_int = g[e[1], e[2]]["intersection"]
        vs = Oscar.vertices(e_int)
        direction = _edge_direction(e_int)
        canonical_direction = _canonical_direction(direction)
        canonical_direction === nothing && continue
        if length(vs) == 1
            ds_with_source = add_d(ds_with_source, vs[1], canonical_direction)
        elseif length(vs) >= 2
            d12 = _normalized_direction(
                [vs[2][1] - vs[1][1], vs[2][2] - vs[1][2]]
            )
            d21 = _normalized_direction(
                [vs[1][1] - vs[2][1], vs[1][2] - vs[2][2]]
            )
            d12 !== nothing && (ds_with_source = add_d(ds_with_source, vs[1], d12))
            d21 !== nothing && (ds_with_source = add_d(ds_with_source, vs[2], d21))
        end
        push!(ds_full, canonical_direction)
    end
    ds_with_source["full"] = ds_full
    return ds_with_source
end

@doc raw"""
    edge_directions(f::Union{Signomial,RationalSignomial};
                    mode::LinearRegionsCalculationMode=OscarMode())

Return normalized boundary directions for the two-dimensional linear regions
of `f`.
"""
function edge_directions(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    if nvars(f) != 2
        throw(
            ArgumentError(
            "edge_directions is only supported for bivariate (2-variable) tropical polynomials or rational maps.",
        ),
        )
    end
    return _edge_directions(get_graph(f; mode = mode))
end

"""
    _vertex_collection(g::MetaGraph)

Return finite vertices and their occurrence counts in stored graph-edge
intersections.
"""
function _vertex_collection(g::MetaGraph)
    vs = Vector{Any}()
    for e in collect(edge_labels(g))
        append!(vs, Oscar.vertices(g[e[1], e[2]]["intersection"]))
    end
    # Count occurrences of each vertex.
    vs_with_mult = Dict()
    for v in vs
        if haskey(vs_with_mult, v)
            vs_with_mult[v] += 1
        else
            vs_with_mult[v] = 1
        end
    end
    return vs_with_mult
end

@doc raw"""
    vertex_collection(f::Union{Signomial,RationalSignomial};
                      mode::LinearRegionsCalculationMode=OscarMode())

Return finite vertices and occurrence counts at linear-region adjacencies.
"""
function vertex_collection(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return vertex_collection(get_graph(f; mode = mode))
end

@doc raw"""
    _vertex_count(g::MetaGraph)

Return the number of distinct finite vertices in graph-edge intersections.
"""
function _vertex_count(g::MetaGraph)
    return length(collect(keys(vertex_collection(g))))
end

@doc raw"""
    vertex_count(f::Union{Signomial,RationalSignomial};
                 mode::LinearRegionsCalculationMode=OscarMode())

Return the number of distinct finite vertices found at region adjacencies.
"""
function vertex_count(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return _vertex_count(get_graph(f; mode = mode))
end
