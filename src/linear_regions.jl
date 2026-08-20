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
    _canonical_halfspace_key(a, rhs)

Return `(hyperplane_key, is_nonpositive_side)` for the halfspace `a*x <= rhs`.
Divide `(a..., -rhs)` by its first nonzero entry to form `hyperplane_key`.
The first nonzero key entry is one. `is_nonpositive_side` is `true` when the
selected halfspace has normalized affine expression at most zero. It is `false`
when the expression is at least zero.
"""
function _canonical_halfspace_key(a, rhs)
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
    _dominance_halfspace_keys(signomials, dominance_indices)

`dominance_indices` specifies one monomial in each signomial. Compare this
monomial with every other monomial in the same signomial. Return the distinct
halfspace keys on which the specified monomials dominate. Each key has the
format returned by `_canonical_halfspace_key`.
"""
function _dominance_halfspace_keys(
        signomials::AbstractVector{<:Signomial}, dominance_indices)
    halfspace_keys = Tuple{Any, Bool}[]

    # Compare each dominant monomial with the other monomials in its signomial.
    for (signomial, dominant_index) in zip(signomials, dominance_indices)
        for competitor_index in Base.eachindex(signomial)
            competitor_index == dominant_index && continue
            A, b = _linear_region_constraints(
                signomial,
                dominant_index,
                OSCAR_POLYHEDRON_COEFF_TYPE;
                competitors = (competitor_index,)
            )
            halfspace_key = _canonical_halfspace_key(@view(A[1, :]), b[1])
            push!(halfspace_keys, halfspace_key)
        end
    end
    return unique!(halfspace_keys)
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
    _intersect_linear_region_all_partitions(partitions; mode, workers=nothing,
                                            base_region=nothing)

Intersect the region partitions in order. Discard an intersection as soon as
it has no interior. If `base_region` is provided, start with that region. The
result then contains only cells inside `base_region`. Use `workers` to test
intersection chunks in parallel when supplied.
"""
function _intersect_linear_region_all_partitions(
        partitions;
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
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
        regions = _intersect_linear_region_partition(
            regions, partition, mode, workers)
        isempty(regions) && break
    end

    return regions
end

"""
    _intersect_linear_region_partition_chunk((region_data, partition_data,
                                               pair_indices, mode,
                                               return_regions))

Test a chunk of region pairs and return their full-dimensional intersections.
If `return_regions` is false, return serializable constraint data instead.
"""
function _intersect_linear_region_partition_chunk(args)
    region_data, partition_data, pair_indices, mode, return_regions = args
    intersections = []
    region_count = length(region_data)
    for pair_index in pair_indices
        partition_index = div(pair_index - 1, region_count) + 1
        region_index = mod(pair_index - 1, region_count) + 1
        index, A_2, b_2 = partition_data[partition_index]
        indices, A_1, b_1 = region_data[region_index]
        A = vcat(A_1, A_2)
        b = vcat(b_1, b_2)
        region = make_polyhedron(A, b; mode = mode)
        if is_full_dimensional(region; mode = mode)
            intersection = if return_regions
                ((indices..., index), region)
            else
                ((indices..., index), A, b)
            end
            push!(intersections, intersection)
        end
    end
    return intersections
end

"""
    _intersect_linear_region_partition(regions, partition, mode, workers)

Refine `regions` with one partition. Use `workers` to test pair chunks in
parallel when supplied.
"""
function _intersect_linear_region_partition(
        regions,
        partition,
        mode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool}
)
    region_data = [(indices, _region_constraint_data(region; mode = mode)...)
                   for (indices, region) in regions]
    partition_data = [(index, _region_constraint_data(region; mode = mode)...)
                      for (index, region) in partition]
    pair_count = Base.Checked.checked_mul(length(partition_data), length(region_data))
    pair_indices = 1:pair_count

    if workers === nothing || pair_count <= 1
        return _intersect_linear_region_partition_chunk(
            (region_data, partition_data, pair_indices, mode, true))
    end

    _assert_tropicalnn_loaded(workers)
    chunks = _index_chunks(pair_count, length(Distributed.workers(workers)))
    caching_pool = Distributed.CachingPool(Distributed.workers(workers))
    try
        chunk_results = Distributed.pmap(caching_pool, chunks) do chunk
            _intersect_linear_region_partition_chunk(
                (region_data, partition_data, chunk, mode, false))
        end
        intersection_data = Base.reduce(vcat, chunk_results)
        return [(indices, make_polyhedron(A, b; mode = mode))
                for (indices, A, b) in intersection_data]
    finally
        Distributed.clear!(caching_pool)
    end
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
        base_region = nothing,
        map_coefficient = _coefficient_rational
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
    partition = _intersect_linear_region_all_partitions(
        dominance_partitions;
        mode = mode,
        workers = workers,
        base_region = base_region
    )

    # Store the constraints and affine map of each cell.
    return map(partition) do (dominance_indices, region)
        A, b = _region_constraint_data(region; mode = mode)
        affine_key = [(map_coefficient(get_coeff(signomial, index)),
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
        denominator_index;
        map_coefficient = _coefficient_rational
)
    coeff = map_coefficient(get_coeff(numerator, numerator_index)) -
            map_coefficient(get_coeff(denominator, denominator_index))
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
    _affine_map_key(matrix, offset)

Return an immutable, globally comparable key for an affine map.
"""
function _affine_map_key(matrix, offset)
    return (size(matrix), Tuple(vec(matrix)), Tuple(offset))
end

"""
    _polyhedral_subdivision(rational_signomials; mode, workers=nothing,
                            base_region=nothing, base_halfspace_keys=[])

Subdivide the domain of `rational_signomials` according to the dominant terms
of their numerators and denominators. Each cell records the supporting
hyperplanes and selected sides from its dominance comparisons. If `base_region`
is provided, subdivide only that region. Add the supporting hyperplanes and
selected sides in `base_halfspace_keys` to each cell.
"""
function _polyhedral_subdivision(
        rational_signomials::AbstractVector{<:RationalSignomial};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
        base_region = nothing,
        base_halfspace_keys = Tuple{Any, Bool}[],
        map_coefficient = _coefficient_rational
)
    numerators = [rational_signomial.num
                  for rational_signomial in rational_signomials]
    denominators = [rational_signomial.den
                    for rational_signomial in rational_signomials]
    # Process denominators first to preserve the existing cell order.
    signomials = vcat(denominators, numerators)

    # Subdivide by all numerator and denominator dominance regions.
    signomial_cells = _signomial_region_partition(
        signomials;
        mode = mode,
        workers = workers,
        base_region = base_region,
        map_coefficient = map_coefficient
    )
    component_count = length(rational_signomials)
    denominator_rows = 1:component_count
    numerator_rows = (component_count + 1):(2 * component_count)

    # Convert each signomial cell to its rational affine map and facet data.
    return map(signomial_cells) do cell
        matrix = @views cell.matrix[numerator_rows, :] -
                        cell.matrix[denominator_rows, :]
        offset = @views cell.offset[numerator_rows] - cell.offset[denominator_rows]
        denominator_indices = cell.data[denominator_rows]
        numerator_indices = cell.data[numerator_rows]
        halfspace_keys = unique!(vcat(
            base_halfspace_keys,
            _dominance_halfspace_keys(numerators, numerator_indices),
            _dominance_halfspace_keys(denominators, denominator_indices)
        ))
        return _Cell(cell.A, cell.b, matrix, offset, halfspace_keys)
    end
end

"""
    _facet_connected_components(cells; mode)

Return one set of cell indices for each full-dimensional region. Two cells
belong to the same region if a sequence of cells joins them. Each consecutive
pair in this sequence must share a facet. The `data` field of each cell must
contain halfspace keys in the format returned by `_canonical_halfspace_key`.
"""
function _facet_connected_components(
        cells::AbstractVector{<:_Cell};
        mode::LinearRegionsCalculationMode
)
    graph = Graphs.SimpleGraph(length(cells))

    # Group cells by each supporting hyperplane and the side they select.
    hyperplane_buckets = Dict{Any, Tuple{Vector{Int}, Vector{Int}}}()
    for (cell_index, cell) in pairs(cells)
        for (hyperplane_key, is_nonpositive_side) in cell.data
            if !haskey(hyperplane_buckets, hyperplane_key)
                hyperplane_buckets[hyperplane_key] = (Int[], Int[])
            end
            nonpositive_cell_indices, nonnegative_cell_indices =
                hyperplane_buckets[hyperplane_key]
            push!(
                is_nonpositive_side ? nonpositive_cell_indices : nonnegative_cell_indices,
                cell_index
            )
        end
    end

    # Collect cell pairs on opposite sides of the same supporting hyperplane.
    pair_hyperplanes = Dict{Tuple{Int, Int}, Vector{Any}}()
    for (hyperplane_key, side_cell_indices) in hyperplane_buckets
        nonpositive_cell_indices, nonnegative_cell_indices = side_cell_indices
        for nonpositive_cell_index in nonpositive_cell_indices,
            nonnegative_cell_index in nonnegative_cell_indices

            # Use one key order to collect all separating hyperplanes.
            # This is later useful: we can directly reject pairs with more than
            # one separating hyperplane!
            cell_pair = minmax(nonpositive_cell_index, nonnegative_cell_index)
            if !haskey(pair_hyperplanes, cell_pair)
                pair_hyperplanes[cell_pair] = Any[]
            end
            push!(pair_hyperplanes[cell_pair], hyperplane_key)
        end
    end

    # Add a graph edge when the cells share a facet.
    for ((left_index, right_index), hyperplane_keys) in pair_hyperplanes
        length(hyperplane_keys) == 1 || continue
        if _cells_share_facet(
            cells[left_index], cells[right_index], only(hyperplane_keys), mode)
            Graphs.add_edge!(graph, left_index, right_index)
        end
    end
    return Graphs.connected_components(graph)
end

"""
    _group_cells(cells; mode)

Group cells by affine-map equality and split each group into full-dimensional
regions. Return the constituent cells and linear regions.
"""
function _group_cells(
        cells::AbstractVector{C};
        mode::LinearRegionsCalculationMode
) where {C <: _Cell}
    isempty(cells) && throw(ArgumentError(
        "No full-dimensional linear regions were found for the rational signomial"
    ))

    # Group cells that define the same affine map.
    map_to_indices = Dict{Any, Vector{Int}}()
    for (index, cell) in pairs(cells)
        key = _affine_map_key(cell.matrix, cell.offset)
        push!(get!(map_to_indices, key, Int[]), index)
    end

    grouped_cells = C[]
    linear_regions = LinearRegion{Cell}[]

    # Split the cells for each affine map into full-dimensional regions.
    for indices in values(map_to_indices)
        affine_cells = cells[indices]
        for region_indices in _facet_connected_components(affine_cells; mode = mode)
            region_cells = affine_cells[region_indices]
            append!(grouped_cells, region_cells)
            push!(linear_regions, LinearRegion(Cell[Cell(cell) for cell in region_cells]))
        end
    end

    return grouped_cells, LinearRegions(linear_regions)
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

    cells = _polyhedral_subdivision(
        q;
        mode = mode,
        workers = workers
    )
    _, regions = _group_cells(cells; mode = mode)
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
