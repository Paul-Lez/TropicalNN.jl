# Compute linear regions one layer at a time.

"""
    _compose_affine(f, matrix, offset)

Compose the signomial `f` with the affine map `x -> matrix * x + offset`.
For a monomial with coefficient `a` and exponent `e`, the new coefficient is
`a + dot(e, offset)`. The new exponent is `transpose(matrix) * e`.
"""
function _compose_affine(
        f::Signomial,
        matrix::AbstractMatrix,
        offset::AbstractVector
)
    exponent_type = eltype(f.exp)
    composed_exponent_type = Base.promote_op(
        *,
        eltype(matrix),
        exponent_type
    )
    composed_exponents = Matrix{composed_exponent_type}(
        undef,
        size(matrix, 2),
        length(f)
    )
    composed_coefficients = similar(f.coeff)
    exponent = Vector{exponent_type}(undef, nvars(f))
    multiplication_exponent = if composed_exponent_type == exponent_type
        exponent
    else
        Vector{composed_exponent_type}(undef, nvars(f))
    end
    transposed_matrix = transpose(matrix)
    coefficient_parent = _coefficient_parent(f)

    # Keep the established column-wise reduction order for floating-point data.
    for index in Base.eachindex(f)
        copyto!(exponent, get_exp(f, index))
        if multiplication_exponent !== exponent
            copyto!(multiplication_exponent, exponent)
        end
        LinearAlgebra.mul!(
            @view(composed_exponents[:, index]),
            transposed_matrix,
            multiplication_exponent
        )
        coefficient = Rational(get_coeff(f, index)) +
                      LinearAlgebra.dot(exponent, offset)
        composed_coefficients[index] = coefficient_parent(Rational{BigInt}(coefficient))
    end

    # Canonical construction merges monomials that composition makes equal.
    return Signomial{composed_exponent_type}(
        composed_exponents,
        composed_coefficients
    )
end

"""
    _compose_affine(layer, matrix, offset)

Compose a rational signomial map with `x -> matrix * x + offset`.
"""
function _compose_affine(
        layer::AbstractVector{<:RationalSignomial},
        matrix::AbstractMatrix,
        offset::AbstractVector
)
    composed_layer = RationalSignomial[]
    sizehint!(composed_layer, length(layer))
    for rational_signomial in layer
        numerator = _compose_affine(
            rational_signomial.num,
            matrix,
            offset
        )
        denominator = _compose_affine(
            rational_signomial.den,
            matrix,
            offset
        )
        push!(composed_layer, RationalSignomial(numerator, denominator))
    end
    return composed_layer
end

"""
    _is_affine_layer(layer)

Return whether every component of `layer` is a quotient of two monomials.
"""
function _is_affine_layer(layer::AbstractVector{<:RationalSignomial})
    return all(layer) do component
        length(component.num) == 1 && length(component.den) == 1
    end
end

"""
    _affine_layer_formula(layer)

Return the matrix and offset of an affine rational signomial map.
"""
function _affine_layer_formula(layer::AbstractVector{<:RationalSignomial})
    component_linear_map_keys = [
        _linear_map_key(component.num, component.den, 1, 1) for component in layer
    ]
    return _affine_formula_from_linear_map_key(component_linear_map_keys)
end

"""
    _layer_subdivision(composed_layer, base_matrix, base_vector, mode, workers)

Subdivide the cell `base_matrix * x <= base_vector` with `composed_layer`.
"""
function _layer_subdivision(
        composed_layer::AbstractVector{<:RationalSignomial},
        base_matrix::AbstractMatrix,
        base_vector::AbstractVector,
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool}
)
    if _is_affine_layer(composed_layer)
        matrix, offset = _affine_layer_formula(composed_layer)
        return [_Cell(
            base_matrix,
            base_vector,
            matrix,
            offset,
            nothing
        )]
    end

    base_region = make_polyhedron(base_matrix, base_vector; mode = mode)
    return _polyhedral_subdivision(
        composed_layer;
        mode = mode,
        workers = workers,
        base_region = base_region
    )
end

"""
    _layer_subdivision_group_job((composed_layer, jobs, mode))

Subdivide one group of base cells that use the same composed source.
"""
function _layer_subdivision_group_job(args)
    composed_layer, jobs, mode = args
    return [
        (cell_index, _layer_subdivision(composed_layer, A, b, mode, nothing))
        for (cell_index, A, b) in jobs
    ]
end

"""Group one composed layer source with the base cells that use it."""
struct _LayerSubdivisionGroup{L}
    source_id::_BoundarySourceID
    composed_layer::L
    cell_indices::Vector{Int}
end

"""
    _apply_affine_layer(cells, layer)

Compose each cell map with `layer`. Each component of `layer` is represented as
a quotient of two monomials.
"""
function _apply_affine_layer(
        cells::AbstractVector{<:_Cell},
        layer::AbstractVector{<:RationalSignomial}
)
    isempty(cells) && return _Cell[]

    layer_matrix, layer_offset = _affine_layer_formula(layer)

    return map(cells) do cell
        _Cell(
            cell.A,
            cell.b,
            layer_matrix * cell.matrix,
            layer_matrix * cell.offset + layer_offset,
            cell.data
        )
    end
end

"""
    _subdivide_cells_by_layer(cells, layer, boundary_sources, stage_id;
                              mode, workers=nothing)

Compose `layer` with each cell map. Subdivide each cell by the resulting
numerator and denominator dominance regions.
"""
function _subdivide_cells_by_layer(
        cells::AbstractVector{<:_Cell},
        layer::AbstractVector{<:RationalSignomial},
        boundary_sources,
        stage_id::Int;
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    # Compose each distinct cell map and give it a stable stage-local ID.
    group_by_map = Dict{Any, _LayerSubdivisionGroup}()
    source_ids = _BoundarySourceID[]
    for (cell_index, cell) in pairs(cells)
        group = get!(group_by_map, _affine_map_key(cell.matrix, cell.offset)) do
            source_id = _BoundarySourceID(stage_id, length(group_by_map) + 1)
            composed_layer = _compose_affine(layer, cell.matrix, cell.offset)
            boundary_sources[source_id] = _boundary_source_components(composed_layer)
            _LayerSubdivisionGroup(source_id, composed_layer, Int[])
        end
        push!(group.cell_indices, cell_index)
        push!(source_ids, group.source_id)
    end

    # Group base cells so each distributed job carries one composed source.
    subdivision_jobs = map(values(group_by_map)) do group
        jobs = [(index, cells[index].A, cells[index].b) for index in group.cell_indices]
        (group.composed_layer, jobs, mode)
    end

    # Subdivide each source group.
    cell_subdivisions = if workers !== nothing && length(cells) > 1
        _assert_tropicalnn_loaded(workers)
        Distributed.pmap(
            _layer_subdivision_group_job,
            workers,
            subdivision_jobs
        )
    else
        [_layer_subdivision_group_job(job) for job in subdivision_jobs]
    end

    # Restore parent-cell order and attach compact provenance locally.
    subdivisions_by_parent = Vector{Any}(undef, length(cells))
    for group_result in cell_subdivisions
        for (cell_index, subdivision) in group_result
            subdivisions_by_parent[cell_index] = subdivision
        end
    end

    next_cells = _Cell[]
    for (parent_cell, source_id, subdivision) in
        zip(cells, source_ids, subdivisions_by_parent)
        for cell in subdivision
            provenance = if cell.data === nothing
                parent_cell.data
            else
                _BoundaryProvenance(source_id, cell.data, parent_cell.data)
            end
            push!(next_cells, _Cell(cell, provenance))
        end
    end
    return next_cells
end

"""
    _linear_regions_composition(layers; mode, workers=nothing)

Return the linear regions of the map obtained by applying `layers` in order.
"""
function _linear_regions_composition(
        layers::AbstractVector{<:AbstractVector{<:RationalSignomial}};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    mode isa _HiGHS && _validate_highs_tolerance(mode.tol)
    input_dimension = nvars(first(first(layers)))
    constraint_type = _linear_region_coefficient_type(mode)
    identity_matrix = Matrix{Rational{BigInt}}(
        LinearAlgebra.I,
        input_dimension,
        input_dimension
    )
    identity_offset = zeros(Rational{BigInt}, input_dimension)

    # Start with the identity map on the full input space.
    current_cells = [
        _Cell(
            zeros(constraint_type, 0, input_dimension),
            constraint_type[],
            identity_matrix,
            identity_offset,
            nothing
        )
    ]
    boundary_sources = Dict{_BoundarySourceID, Any}()

    for (stage_id, layer) in pairs(layers)
        if _is_affine_layer(layer)
            current_cells = _apply_affine_layer(current_cells, layer)
        else
            current_cells = _subdivide_cells_by_layer(
                current_cells,
                layer,
                boundary_sources,
                stage_id;
                mode = mode,
                workers = workers
            )
        end
    end

    # Materialize the final regions and join their facet-connected cells.
    _, regions = _group_cells(current_cells, boundary_sources; mode = mode)
    return regions
end

"""
    linear_regions(layers; mode, workers=nothing)

Return the linear regions for a sequence of rational signomial maps. Each vector
in `layers` defines one map. The function applies the maps in order. It does not
construct their complete composition.
"""
function linear_regions(
        layers::AbstractVector{<:AbstractVector{<:RationalSignomial}};
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    isempty(layers) && throw(ArgumentError("Layer vector must have at least one layer"))
    for layer in layers
        isempty(layer) &&
            throw(ArgumentError("RationalSignomial vector must have at least one component"))
        any(rational_signomial -> length(rational_signomial.num) == 0, layer) &&
            throw(ArgumentError("RationalSignomial numerator must have at least one monomial"))
        any(rational_signomial -> length(rational_signomial.den) == 0, layer) &&
            throw(ArgumentError("RationalSignomial denominator must have at least one monomial"))
    end

    return _linear_regions_composition(layers; mode = mode, workers = workers)
end

"""
    linear_regions(linear_maps, biases, thresholds=nothing;
                   mode, workers=nothing)

Return the linear regions of an MLP. Each hidden layer applies
`max.(A * x + b, threshold)`. The final layer is affine. The function computes
the regions after each layer. It does not construct one rational signomial map
for the complete network.
"""
function linear_regions(
        linear_maps::AbstractVector{<:AbstractMatrix},
        biases::AbstractVector{<:AbstractVector},
        thresholds::Union{Nothing, AbstractVector{<:AbstractVector}} = nothing;
        mode::LinearRegionsCalculationMode,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing
)
    layers = _mlp_to_tropical_layers(linear_maps, biases, thresholds)
    return _linear_regions_composition(
        layers;
        mode = mode,
        workers = workers
    )
end
