# Shared implementation for computing linear regions of tropical Puiseux
# polynomials and rational functions.

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
    _signomial_region_data_job((signomial, mode))

Compute the dominance regions of one signomial on a worker process.
"""
function _signomial_region_data_job(args)
    signomial, mode = args
    return _linear_region_data_parallel(signomial, mode, nothing)
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
    mode isa _HiGHS && _validate_highs_tolerance(mode.tol)
    monomial_count = length(signomial)

    # Use current process if there is no work to distribute
    # i.e. we have no workers or only one monomial to process.
    if workers === nothing || monomial_count <= 1
        return [_linear_region_data((signomial, index, mode))
                for index in Base.eachindex(signomial)]
    end

    # Divide the dominance region computations among the available workers.
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
    _intersect_linear_region_partitions(partitions; mode, base_region=nothing)

Intersect the region partitions in order. Discard an intersection as soon as
it has no interior. If `base_region` is provided, start with that region. The
result then contains only cells inside `base_region`.
"""
function _intersect_linear_region_partitions(
        partitions;
        mode::LinearRegionsCalculationMode,
        base_region = nothing
)
    # Select the initial regions to partition by and the partitions that remain.
    first_partition = first(partitions)

    if base_region === nothing
        # if there's no base region, then we start with the first partition as the
        # set of "candidate regions" that will be refined by the later partitions
        regions = [((index,), region) for (index, region) in first_partition]
        remaining_partitions = Iterators.drop(partitions, 1)
    else
        # if we've been given a base region then we start with that
        # and will refine it incrementally with _all_ the partitions, including the first one
        regions = [((), base_region)]
        remaining_partitions = partitions
    end

    # Refine each candidate region with the next partition.
    for partition in remaining_partitions
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
    _signomial_region_partition(signomials; mode, workers=nothing,
                                base_region=nothing)

Return the common subdivision of `signomials` into dominance regions. Each cell
records the dominant monomial in every signomial. If `base_region` is provided,
subdivide only that region.
"""
function _signomial_region_partition(
        signomials::AbstractVector{<:Signomial};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
        base_region = nothing
)
    isempty(signomials) && return _Cell[]

    # Compute the dominance regions of each component signomial.
    if workers !== nothing && length(signomials) > 1
        _assert_tropicalnn_loaded(workers)
        region_data_by_signomial = Distributed.pmap(
            _signomial_region_data_job,
            workers,
            [(signomial, mode) for signomial in signomials]
        )
    else
        region_data_by_signomial = [_linear_region_data_parallel(signomial, mode, workers)
                                    for signomial in signomials]
    end

    # Construct the polyhedral dominance partition of each signomial.
    dominance_partitions = map(region_data_by_signomial) do region_data
        return [(monomial_index, make_polyhedron(data[1], data[2]; mode = mode))
                for (monomial_index, data) in pairs(region_data) if data[3]]
    end

    # Intersect the partitions to obtain a common subdivision.
    partition = _intersect_linear_region_partitions(
        dominance_partitions;
        mode = mode,
        base_region = base_region
    )
    isempty(partition) && return _Cell[]

    # Store the constraints and affine map of each cell.
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

"""
    _affine_formula_from_linear_map_key(key)

Convert a rational-subdivision key into the matrix and offset of its affine
map. The key has one `(offset, coefficients)` pair for each output coordinate.
The coefficients are the corresponding row of the affine matrix.
"""
function _affine_formula_from_linear_map_key(key)
    rows = [permutedims(collect(component[2])) for component in key]
    matrix = reduce(vcat, rows)
    offset = [component[1] for component in key]
    return matrix, offset
end

"""
    _rational_region_intersections_chunk((numerator_cells, denominator_cells,
                                          candidate_pairs, mode))

Test the requested numerator/denominator partition pairs for full-dimensional
intersection. Return an internal cell for each accepted intersection. The input
is a tuple so we can directly pass this to `pmap`.
"""
function _rational_region_intersections_chunk(args)
    numerator_cells, denominator_cells, candidate_pairs, mode = args
    cells = _Cell[]

    # Intersect each requested pair of numerator and denominator cells.
    for (numerator_cell_index, denominator_cell_index) in candidate_pairs
        numerator_cell = numerator_cells[numerator_cell_index]
        denominator_cell = denominator_cells[denominator_cell_index]
        intersection_matrix = vcat(numerator_cell.A, denominator_cell.A)
        intersection_vector = vcat(numerator_cell.b, denominator_cell.b)
        region = make_polyhedron(intersection_matrix, intersection_vector; mode = mode)

        # Store each full-dimensional intersection and its compact selection.
        if is_full_dimensional(region; mode = mode)
            push!(
                cells,
                _Cell(
                    intersection_matrix,
                    intersection_vector,
                    numerator_cell.matrix - denominator_cell.matrix,
                    numerator_cell.offset - denominator_cell.offset,
                    (numerator_cell.data..., denominator_cell.data...)
                )
            )
        end
    end

    return cells
end

"""
    _rational_region_intersections_parallel(numerator_cells, denominator_cells,
                                            mode, workers)

Test every numerator/denominator partition pair for full-dimensional
intersection. Use `workers` to evaluate pair chunks in parallel when supplied.
"""
function _rational_region_intersections_parallel(
        numerator_cells::AbstractVector{<:_Cell},
        denominator_cells::AbstractVector{<:_Cell},
        mode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool}
)
    # Create all pairs of numerator and denominator cells.
    candidate_pairs = Tuple{Int, Int}[]
    for numerator_cell_index in Base.eachindex(numerator_cells)
        for denominator_cell_index in Base.eachindex(denominator_cells)
            push!(candidate_pairs, (numerator_cell_index, denominator_cell_index))
        end
    end

    # Process the pairs locally if there is no worker pool or at most one pair.
    if workers === nothing || length(candidate_pairs) <= 1
        return _rational_region_intersections_chunk(
            (
                numerator_cells,
                denominator_cells,
                candidate_pairs,
                mode
            )
        )
    end

    # Divide the cell pairs among the available workers.
    _assert_tropicalnn_loaded(workers)
    chunks = _index_chunks(
        length(candidate_pairs), length(Distributed.workers(workers)))
    pair_chunks = [candidate_pairs[chunk] for chunk in chunks]
    chunk_results = Distributed.pmap(
        _rational_region_intersections_chunk,
        workers,
        [(numerator_cells,
             denominator_cells,
             pair_chunk,
             mode)
         for pair_chunk in pair_chunks]
    )
    return Base.reduce(vcat, chunk_results)
end

"""
    _polyhedral_subdivision(rational_signomials; mode, workers=nothing,
                            base_region=nothing)

Subdivide the domain of `rational_signomials` according to the dominant terms
of their numerators and denominators. If `base_region` is provided, subdivide
only that region.
"""
function _polyhedral_subdivision(
        rational_signomials::AbstractVector{<:RationalSignomial};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
        base_region = nothing
)
    # Separate the numerator and denominator signomials.
    numerators = [rational_signomial.num
                  for rational_signomial in rational_signomials]
    denominators = [rational_signomial.den
                    for rational_signomial in rational_signomials]

    # Compute the common subdivisions of the numerator and denominator signomials.
    numerator_cells = _signomial_region_partition(
        numerators;
        mode = mode,
        workers = workers,
        base_region = base_region
    )
    denominator_cells = _signomial_region_partition(
        denominators;
        mode = mode,
        workers = workers,
        base_region = base_region
    )

    # Intersect the numerator and denominator subdivisions.
    return _rational_region_intersections_parallel(
        numerator_cells,
        denominator_cells,
        mode,
        workers
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
    isempty(q) &&
        throw(ArgumentError("RationalSignomial vector must have at least one component"))
    any(rational_signomial -> length(rational_signomial.num) == 0, q) &&
        throw(ArgumentError("RationalSignomial numerator must have at least one monomial"))
    any(rational_signomial -> length(rational_signomial.den) == 0, q) &&
        throw(ArgumentError("RationalSignomial denominator must have at least one monomial"))

    source_id = _BoundarySourceID(1, 1)
    boundary_sources = Dict(source_id => _boundary_source_components(q))
    cells = _polyhedral_subdivision(q; mode = mode, workers = workers)
    if !isempty(cells)
        cells = _attach_boundary_provenance(cells, source_id)
    end
    _, regions = _group_cells(cells, boundary_sources; mode = mode)
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
    return linear_regions([q]; mode = mode, workers = workers)
end
