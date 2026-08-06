# Experimental layerwise linear-region propagation.

"""
    _LabelledAffinePiece

A polyhedral piece in the original input space together with its exact affine
formula where the input data permits exact arithmetic. `A` and `b` are kept
alongside `region` so Oscar pullbacks never pass through the floating-point
`get_matrix` compatibility API.
"""
struct _LabelledAffinePiece{R, AM, BV, MM, OV, BM}
    region::R
    A::AM
    b::BV
    matrix::MM
    offset::OV
    boundaries::BM
end

"""
    _LabelledAffinePiece(region, A, b, matrix, offset)

Construct a labelled piece and derive boundary provenance from its constraints.
"""
function _LabelledAffinePiece(region, A, b, matrix, offset)
    return _LabelledAffinePiece(
        region,
        A,
        b,
        matrix,
        offset,
        _boundary_sides_from_constraints(A, b)
    )
end

"""
    _affine_formula_from_linear_map_key(key)

Convert the rational-subdivision key into the matrix and offset of its affine
map without evaluating or sampling the function.
"""
function _affine_formula_from_linear_map_key(key)
    isempty(key) && throw(ArgumentError("An affine map must have at least one output"))
    rows = [permutedims(collect(component[2])) for component in key]
    matrix = reduce(vcat, rows)
    offset = [component[1] for component in key]
    return matrix, offset
end

"""
    _affine_pullback_signomial(f, matrix, offset)

Pull the signomial `f` back through `y = matrix * x + offset`. A monomial with
coefficient `a` and exponent `e` becomes the monomial with coefficient
`a + dot(e, offset)` and exponent `transpose(matrix) * e`. This operation is
affine substitution only: it never multiplies monomial sets, and therefore
cannot cause the global-composition expansion avoided by this algorithm.
"""
function _affine_pullback_signomial(f::Signomial, matrix, offset)
    size(matrix, 1) == nvars(f) || throw(DimensionMismatch(
        "Affine pullback map has output dimension $(size(matrix, 1)), " *
        "but the signomial expects $(nvars(f)) variables"
    ))
    length(offset) == size(matrix, 1) || throw(DimensionMismatch(
        "Affine pullback map has $(size(matrix, 1)) outputs but offset length " *
        "$(length(offset))"
    ))

    coefficient_parent = _coefficient_parent(f)
    pulled_exponents = [collect(transpose(matrix) * collect(get_exp(f, i)))
                        for i in Base.eachindex(f)]
    pulled_coefficients = Oscar.TropicalSemiringElem{typeof(max)}[coefficient_parent(_constraint_scalar(
                                                                      OSCAR_POLYHEDRON_COEFF_TYPE,
                                                                      Rational(get_coeff(f, i)) +
                                                                      LinearAlgebra.dot(
                                                                          collect(get_exp(f, i)),
                                                                          offset)
                                                                  ))
                                                                  for i in Base.eachindex(f)]

    # Canonicalization is important for rank-deficient prefixes: monomials
    # that become identical on the prefix image are merged exactly here.
    return Signomial(pulled_coefficients, pulled_exponents; sorted = false)
end

"""
    _affine_map_key(matrix, offset)

Return an immutable, globally comparable key for an affine map.
"""
function _affine_map_key(matrix, offset)
    return (size(matrix), Tuple(vec(matrix)), Tuple(offset))
end

"""
    _row_matches_boundary(row, rhs, boundary_key; atol=1e-9)

Return whether a floating-point inequality has the specified boundary.
"""
function _row_matches_boundary(row, rhs, boundary_key; atol = 1.0e-9)
    equation = [Float64.(row); -Float64(rhs)]
    pivot_index = findfirst(value -> abs(value) > atol, equation)
    pivot_index === nothing && return false
    normalized = equation ./ equation[pivot_index]
    target = Float64.(collect(boundary_key))
    length(normalized) == length(target) || return false
    scale = max(1.0, maximum(abs, target))
    return maximum(abs, normalized .- target) <= atol * scale
end

"""
    _highs_regions_share_boundary_facet(left, right, boundary_key, mode)

Test whether two full-dimensional pieces meet in the relative interior of the
candidate boundary. Boundary rows themselves are imposed as one equality;
all other inequalities receive a common positive normalized slack. Thus one
LP replaces the general codimension routine's feasibility, inflation, and
per-row implicit-equality checks.
"""
function _highs_regions_share_boundary_facet(
        left::_LabelledAffinePiece,
        right::_LabelledAffinePiece,
        boundary_key,
        mode::_HiGHS
)
    n = size(left.A, 2)
    length(boundary_key) == n + 1 || return regions_intersect_codimension_le_one(
        left.region,
        right.region;
        mode = mode
    )
    normal = Float64.(collect(boundary_key[1:n]))
    rhs = -Float64(boundary_key[n + 1])
    LinearAlgebra.norm(normal) > 0 || return false

    model = create_highs_model(; solver = mode.solver, threads = mode.threads)
    @variable(model, x[1:n])
    @variable(model, epsilon)
    @constraint(model, LinearAlgebra.dot(normal, x) == rhs)
    @constraint(model, epsilon <= 1)

    for (A, b) in ((left.A, left.b), (right.A, right.b))
        for i in axes(A, 1)
            row = @view A[i, :]
            if all(value -> abs(Float64(value)) <= 1.0e-12, row)
                b[i] < 0 && return false
                continue
            end
            _row_matches_boundary(row, b[i], boundary_key) && continue
            row_norm = LinearAlgebra.norm(Float64.(row))
            normalized_row = Float64.(row) ./ row_norm
            normalized_rhs = Float64(b[i]) / row_norm
            @constraint(model,
                LinearAlgebra.dot(normalized_row, x) + epsilon <= normalized_rhs)
        end
    end
    @objective(model, Max, epsilon)
    return _highs_has_positive_slack(
        model,
        epsilon,
        mode.tol;
        context = "HiGHS facet check"
    )
end

"""
    _regions_share_boundary_facet(left, right, boundary_key, mode)

Test facet adjacency on a known candidate boundary using the selected backend.
"""
function _regions_share_boundary_facet(left, right, boundary_key, mode)
    if mode isa _HiGHS
        return _highs_regions_share_boundary_facet(left, right, boundary_key, mode)
    end
    return regions_intersect_codimension_le_one(
        left.region,
        right.region;
        mode = mode
    )
end

"""
    _labelled_piece_components(pieces; mode)

Return index sets for components formed by codimension-at-most-one adjacency.
Indices, rather than region objects, are graph vertices so repeated equivalent
polyhedra remain well-defined.
"""
function _labelled_piece_components(
        pieces::Vector{_LabelledAffinePiece};
        mode::LinearRegionsCalculationMode
)
    length(pieces) <= 1 && return [collect(Base.eachindex(pieces))]

    graph = Graphs.SimpleGraph(length(pieces))
    boundary_buckets = Dict{Any, Tuple{Vector{Int}, Vector{Int}}}()
    for (piece_index, piece) in pairs(pieces)
        for (key, side) in piece.boundaries
            negative, positive = get!(boundary_buckets, key) do
                (Int[], Int[])
            end
            push!(side ? positive : negative, piece_index)
        end
    end

    pair_boundaries = Dict{Tuple{Int, Int}, Vector{Any}}()
    for (key, (negative, positive)) in boundary_buckets
        for i in negative, j in positive

            i == j && continue
            pair = minmax(i, j)
            push!(get!(pair_boundaries, pair, Any[]), key)
        end
    end

    for ((i, j), boundary_keys) in pair_boundaries
        # Opposing constraints on two distinct hyperplanes force an
        # intersection of codimension at least two, so they cannot glue.
        length(boundary_keys) == 1 || continue
        if _regions_share_boundary_facet(pieces[i], pieces[j], only(boundary_keys), mode)
            Graphs.add_edge!(graph, i, j)
        end
    end
    return Graphs.connected_components(graph)
end

"""
    _group_labelled_affine_pieces(pieces; mode)

Group pieces globally by affine-map equality and split every group using the
package adjacency convention. Return all constituent pieces, the corresponding
`LinearRegions`, the number of affine-map groups, and the number of components.
"""
function _group_labelled_affine_pieces(
        pieces::Vector{_LabelledAffinePiece};
        mode::LinearRegionsCalculationMode
)
    isempty(pieces) && throw(ArgumentError(
        "No full-dimensional pullback pieces survived layer propagation"
    ))

    map_to_pieces = Dict{Any, Vector{_LabelledAffinePiece}}()
    for piece in pieces
        key = _affine_map_key(piece.matrix, piece.offset)
        push!(get!(map_to_pieces, key, _LabelledAffinePiece[]), piece)
    end

    region_type = typeof(first(pieces).region)
    grouped_pieces = _LabelledAffinePiece[]
    linear_region_vec = LinearRegion{region_type}[]
    component_count = 0

    for affine_pieces in values(map_to_pieces)
        for component_indices in _labelled_piece_components(affine_pieces; mode = mode)
            component_count += 1
            component = affine_pieces[component_indices]
            append!(grouped_pieces, component)
            regions = region_type[piece.region for piece in component]
            push!(linear_region_vec, LinearRegion(regions))
        end
    end

    return (
        grouped_pieces,
        LinearRegions(linear_region_vec),
        length(map_to_pieces),
        component_count
    )
end

"""
    _validated_composition_layers(layers)

Validate layer maps and return concretely typed rational-signomial vectors.
"""
function _validated_composition_layers(layers)
    layers isa AbstractVector ||
        throw(ArgumentError("layers must be a vector of rational-signomial vectors"))
    isempty(layers) && throw(ArgumentError(
        "Layer composition requires at least one layer"
    ))

    validated = Vector{Vector{RationalSignomial}}(undef, length(layers))
    previous_output_dim = nothing
    for (k, layer) in pairs(layers)
        layer isa AbstractVector ||
            throw(ArgumentError("Layer $k must be a vector of RationalSignomials"))
        isempty(layer) && throw(ArgumentError("Layer $k must have at least one output"))
        all(q -> q isa RationalSignomial, layer) ||
            throw(ArgumentError("Layer $k contains a non-RationalSignomial component"))

        typed_layer = RationalSignomial[q for q in layer]
        input_dim = _validate_rational_signomial_vector(
            typed_layer;
            context = "Layer $k"
        )
        if previous_output_dim !== nothing && input_dim != previous_output_dim
            throw(DimensionMismatch(
                "Layer $k expects input dimension $input_dim, but layer $(k - 1) " *
                "has output dimension $previous_output_dim"
            ))
        end
        validated[k] = typed_layer
        previous_output_dim = length(typed_layer)
    end
    return validated
end

"""
    _prefix_subdivision(pulled_layer, prefix_A, prefix_b, prefix_boundaries,
                        mode, workers)

Compute one prefix-conditioned layer subdivision and return transportable
constraint data with its statistics.
"""
function _prefix_subdivision(
        pulled_layer,
        prefix_A,
        prefix_b,
        prefix_boundaries,
        mode,
        workers
)
    base_region = make_polyhedron(prefix_A, prefix_b; mode = mode)
    atomic_data, stats = _rational_atomic_subdivision(
        pulled_layer;
        mode = mode,
        workers = workers,
        full_dimensional_only = true,
        base_region = base_region,
        base_boundaries = prefix_boundaries,
        return_boundary_data = true,
        return_stats = true
    )
    cells = [(key, A, b, boundaries)
             for (key, _, A, b, boundaries) in atomic_data]
    return cells, stats
end

"""
    _prefix_subdivision_chunk((jobs, mode))

Evaluate a chunk of prefix-conditioned subdivisions on one worker.
"""
function _prefix_subdivision_chunk(args)
    jobs, mode = args
    return [_prefix_subdivision(job..., mode, nothing) for job in jobs]
end

"""
    _linear_regions_composition(layers; mode, workers=nothing)

Internal implementation returning both linear regions and per-layer statistics.
"""
function _linear_regions_composition(
        layers;
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    mode isa _HiGHS && _validate_highs_tolerance(mode.tol)
    validated_layers = _validated_composition_layers(layers)
    input_dim = nvars(first(first(validated_layers)).num)
    constraint_type = _linear_region_coefficient_type(mode)
    identity_matrix = Matrix{Rational{BigInt}}(LinearAlgebra.I, input_dim, input_dim)
    identity_offset = zeros(Rational{BigInt}, input_dim)
    initial_A = zeros(constraint_type, 0, input_dim)
    initial_b = constraint_type[]
    initial_region = make_polyhedron(initial_A, initial_b; mode = mode)
    prefix_pieces = _LabelledAffinePiece[
        _LabelledAffinePiece(
        initial_region,
        initial_A,
        initial_b,
        identity_matrix,
        identity_offset
    )
    ]

    stats = NamedTuple[]
    current_regions = nothing
    for (k, layer) in pairs(validated_layers)
        started = time_ns()
        input_piece_count = length(prefix_pieces)
        affine_layer = all(q -> length(q.num) == 1 && length(q.den) == 1, layer)
        pulled_layer_cache = Dict{Any, Vector{RationalSignomial}}()
        candidates = _LabelledAffinePiece[]
        candidate_count = 0
        partial_intersection_count = 0
        rational_pair_count = 0
        numerator_cell_count = 0
        denominator_cell_count = 0

        prepared_prefixes = map(prefix_pieces) do prefix
            prefix_key = _affine_map_key(prefix.matrix, prefix.offset)
            pulled_layer = get!(pulled_layer_cache, prefix_key) do
                RationalSignomial[RationalSignomial(
                                      _affine_pullback_signomial(q.num, prefix.matrix, prefix.offset),
                                      _affine_pullback_signomial(q.den, prefix.matrix, prefix.offset)
                                  ) for q in layer]
            end
            (prefix, pulled_layer)
        end

        if affine_layer
            for (prefix, pulled_layer) in prepared_prefixes
                key = [_linear_map_key(q.num, q.den, 1, 1) for q in pulled_layer]
                composite_matrix,
                composite_offset = _affine_formula_from_linear_map_key(key)
                push!(candidates,
                    _LabelledAffinePiece(
                        prefix.region,
                        prefix.A,
                        prefix.b,
                        composite_matrix,
                        composite_offset,
                        prefix.boundaries
                    ))
                candidate_count += 1
                rational_pair_count += 1
                numerator_cell_count += 1
                denominator_cell_count += 1
            end
        else
            subdivision_results = if workers !== nothing && length(prepared_prefixes) > 1
                _assert_tropicalnn_loaded(workers)
                # Coarse prefix chunks avoid nested `pmap` calls for individual
                # two-monomial dominance checks.
                jobs = [(pulled_layer, prefix.A, prefix.b, prefix.boundaries)
                        for (prefix, pulled_layer) in prepared_prefixes]
                chunks = _index_chunks(
                    length(jobs),
                    length(Distributed.workers(workers))
                )
                chunk_results = Distributed.pmap(
                    _prefix_subdivision_chunk,
                    workers,
                    [(jobs[chunk], mode) for chunk in chunks]
                )
                Base.reduce(vcat, chunk_results)
            else
                [_prefix_subdivision(
                     pulled_layer,
                     prefix.A,
                     prefix.b,
                     prefix.boundaries,
                     mode,
                     workers
                 ) for (prefix, pulled_layer) in prepared_prefixes]
            end

            for (atomic_data, local_stats) in subdivision_results
                candidate_count += local_stats.candidates_tested
                partial_intersection_count += local_stats.partial_intersections_tested
                rational_pair_count += local_stats.rational_pairs_tested
                numerator_cell_count += local_stats.numerator_cells
                denominator_cell_count += local_stats.denominator_cells

                for (key, A, b, boundaries) in atomic_data
                    composite_matrix,
                    composite_offset = _affine_formula_from_linear_map_key(key)
                    push!(
                        candidates,
                        _LabelledAffinePiece(
                            make_polyhedron(A, b; mode = mode),
                            A,
                            b,
                            composite_matrix,
                            composite_offset,
                            boundaries
                        )
                    )
                end
            end
        end

        prefix_pieces, current_regions, group_count,
        component_count = _group_labelled_affine_pieces(candidates; mode = mode)
        push!(stats,
            (
                layer = k,
                layer_cells = length(candidates),
                pullback_candidates_tested = candidate_count,
                full_dimensional_candidates_retained = length(candidates),
                affine_map_groups = group_count,
                glued_components = component_count,
                constituent_pieces = length(prefix_pieces),
                input_constituent_pieces = input_piece_count,
                unique_prefix_affine_maps = length(pulled_layer_cache),
                numerator_cells = numerator_cell_count,
                denominator_cells = denominator_cell_count,
                partial_intersections_tested = partial_intersection_count,
                rational_pairs_tested = rational_pair_count,
                affine_layer_fast_path = affine_layer,
                subdivision_strategy = :prefix_conditioned,
                elapsed_seconds = (time_ns() - started) / 1.0e9
            ))
    end

    return current_regions::LinearRegions, stats
end

"""
    linear_regions(layers; mode, workers=nothing)

Compute maximal linear regions of a composition of vector-valued rational
signomial layers without constructing the globally composed rational map.
Layer cells are pulled back through labelled affine prefix pieces in the
original input space. Each entry of `layers` is one vector-valued layer map.
"""
function linear_regions(
        layers::AbstractVector{<:AbstractVector{<:RationalSignomial}};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    regions, _ = _linear_regions_composition(layers; mode = mode, workers = workers)
    return regions
end

"""
    linear_regions(linear_maps, biases, thresholds=nothing;
                   mode, workers=nothing, return_stats=false)

Convert each MLP layer independently and propagate labelled affine pieces by
pullback. Hidden layers apply `max.(A*x + b, threshold)` and the final layer is
affine. No globally composed rational expression is constructed. With
`return_stats=true`, return `(regions, stats)`, where `stats` contains cell,
candidate, affine-group, component, and elapsed-time counts for every layer.
"""
function linear_regions(
        linear_maps::AbstractVector{<:AbstractMatrix},
        biases::AbstractVector{<:AbstractVector},
        thresholds::Union{Nothing, AbstractVector{<:AbstractVector}} = nothing;
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
        return_stats::Bool = false
)
    layers = _mlp_to_tropical_layers(linear_maps, biases, thresholds)
    regions, stats = _linear_regions_composition(
        layers;
        mode = mode,
        workers = workers
    )
    return return_stats ? (regions, stats) : regions
end
