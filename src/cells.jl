# Data types for affine cells and linear regions.

"""
    _AbstractCell

Common internal interface for public and internal affine cells.
"""
abstract type _AbstractCell end

@doc raw"""
    Cell(A, b, matrix, offset)

One affine cell of a tropical signomial or rational signomial. The inequalities
`A * x <= b` define the cell. On the cell, the function is
`matrix * x + offset`.
"""
struct Cell{AM, BV, MM, OV} <: _AbstractCell
    A::AM
    b::BV
    matrix::MM
    offset::OV
end

"""Stable stage and stage-local identifier for one boundary source."""
struct _BoundarySourceID
    stage_id::Int
    local_id::Int
end

"""
    _BoundaryProvenance(source_id, selection, parent)

Link one compact dominance selection to a stable source ID. The selection lists
the numerator components before the denominator components. `parent` records
the subdivision that contains this cell.
"""
struct _BoundaryProvenance{I, P}
    source_id::_BoundarySourceID
    selection::I
    parent::P
end

"""Unordered monomial selection change for one source component."""
struct _MonomialTransition
    source_id::_BoundarySourceID
    component_id::Int
    lower_index::Int
    upper_index::Int
end

"""
    _Cell(A, b, matrix, offset, data)

Store an affine cell with data used by an internal computation. For a
signomial subdivision, `data` contains the dominant monomial indices. For a
rational subdivision, it contains compact boundary provenance. The constraint
matrix and vector have the same coefficient type.
"""
struct _Cell{
    D,
    T,
    AM <: AbstractMatrix{T},
    BV <: AbstractVector{T},
    MM,
    OV
} <: _AbstractCell
    A::AM
    b::BV
    matrix::MM
    offset::OV
    data::D
end

"""
    Cell(cell::_Cell)

Return the public cell data and omit internal computation data.
"""
Cell(cell::_Cell) = Cell(cell.A, cell.b, cell.matrix, cell.offset)
_Cell(cell::_Cell, data) = _Cell(cell.A, cell.b, cell.matrix, cell.offset, data)
_attach_boundary_provenance(cells, source_id, parent = nothing) =
    [_Cell(cell, _BoundaryProvenance(source_id, cell.data, parent))
     for cell in cells]

_boundary_source_components(q) = vcat(getproperty.(q, :num), getproperty.(q, :den))

@doc raw"""
    LinearRegion{C}

One linear region of a tropical signomial or rational signomial. A linear
region can contain more than one affine cell.
"""
struct LinearRegion{C <: Cell}
    cells::Vector{C}
end

Base.length(lr::LinearRegion) = length(lr.cells)
Base.iterate(lr::LinearRegion) = iterate(lr.cells)
Base.iterate(lr::LinearRegion, state) = iterate(lr.cells, state)
Base.getindex(lr::LinearRegion, i::Int) = lr.cells[i]

@doc raw"""
    LinearRegions{C}

Result of `linear_regions`. Each element is a `LinearRegion`.
"""
struct LinearRegions{C <: Cell}
    regions::Vector{LinearRegion{C}}
end

Base.length(lrs::LinearRegions) = length(lrs.regions)
Base.iterate(lrs::LinearRegions) = iterate(lrs.regions)
Base.iterate(lrs::LinearRegions, state) = iterate(lrs.regions, state)
Base.getindex(lrs::LinearRegions, i::Int) = lrs.regions[i]

_affine_map_key(matrix, offset) = (size(matrix), Tuple(vec(matrix)), Tuple(offset))

"""
    _index_component_selections!(buckets, provenance, cell_index)

Index the selections in a provenance chain by source component.
"""
function _index_component_selections!(
        buckets,
        provenance::_BoundaryProvenance,
        cell_index::Int
)
    current = provenance
    while current !== nothing
        for (component_id, monomial_index) in pairs(current.selection)
            source_component = (current.source_id, component_id)
            selection_buckets = get!(buckets, source_component) do
                Dict{Int, Vector{Int}}()
            end
            push!(get!(selection_buckets, monomial_index, Int[]), cell_index)
        end
        current = current.parent
    end
    return buckets
end

"""
    _candidate_pair_transitions(cells, boundary_sources, transition_cache, mode)

Return candidate cell pairs and all transition witnesses for each pair.
"""
function _candidate_pair_transitions(
        cells::AbstractVector{<:_Cell},
        boundary_sources,
        transition_cache,
        mode
)
    component_selections = Dict{Tuple{_BoundarySourceID, Int}, Dict{Int, Vector{Int}}}()
    for (cell_index, cell) in pairs(cells)
        cell.data isa _BoundaryProvenance || continue
        _index_component_selections!(component_selections, cell.data, cell_index)
    end

    pair_transitions = Dict{Tuple{Int, Int}, Vector{_MonomialTransition}}()
    for ((source_id, component_id), selection_cells) in component_selections
        length(selection_cells) <= 1 && continue
        signomial = boundary_sources[source_id][component_id]

        transitions = Set{Tuple{Int, Int}}()
        for selected_index in keys(selection_cells)
            cache_key = (source_id, component_id, selected_index)
            adjacent_indices = get!(transition_cache, cache_key) do
                _dominance_transition_indices(signomial, selected_index, mode)
            end
            for adjacent_index in adjacent_indices
                haskey(selection_cells, adjacent_index) || continue
                push!(transitions, minmax(selected_index, adjacent_index))
            end
        end

        for (lower_index, upper_index) in transitions
            transition = _MonomialTransition(source_id, component_id, lower_index, upper_index)
            for left_index in selection_cells[lower_index],
                right_index in selection_cells[upper_index]
                left_index == right_index && continue
                cell_pair = minmax(left_index, right_index)
                push!(get!(pair_transitions, cell_pair, _MonomialTransition[]), transition)
            end
        end
    end
    return pair_transitions
end

"""
    _facet_connected_components(cells, boundary_sources, transition_cache; mode)

Return one index set for each facet-connected component of `cells`.
"""
function _facet_connected_components(
        cells::AbstractVector{<:_Cell},
        boundary_sources,
        transition_cache;
        mode
)
    length(cells) <= 1 && return [collect(Base.eachindex(cells))]

    graph = Graphs.SimpleGraph(length(cells))
    pair_transitions = _candidate_pair_transitions(
        cells,
        boundary_sources,
        transition_cache,
        mode
    )

    for ((left_index, right_index), transitions) in pair_transitions
        if _cells_share_facet(
            cells[left_index],
            cells[right_index],
            transitions,
            boundary_sources,
            mode
        )
            Graphs.add_edge!(graph, left_index, right_index)
        end
    end
    return Graphs.connected_components(graph)
end

"""
    _group_cells(cells, boundary_sources; mode)

Group cells by affine-map equality and split each group into full-dimensional
regions. Return the constituent cells and linear regions.
"""
function _group_cells(
        cells::AbstractVector{C},
        boundary_sources;
        mode
) where {C <: _Cell}
    isempty(cells) && throw(ArgumentError(
        "No full-dimensional linear regions were found for the rational signomial"
    ))

    # Group cells that define the same affine map.
    map_to_indices = Dict{Any, Vector{Int}}()
    for (index, cell) in pairs(cells)
        push!(get!(map_to_indices,
            _affine_map_key(cell.matrix, cell.offset), Int[]), index)
    end

    grouped_cells = C[]
    linear_regions = LinearRegion{Cell}[]
    transition_cache = Dict{Tuple{_BoundarySourceID, Int, Int}, Vector{Int}}()

    # Split repeated affine-map groups into full-dimensional regions.
    for indices in values(map_to_indices)
        affine_cells = cells[indices]
        components = if length(indices) == 1
            (1:1,)
        else
            _facet_connected_components(
                affine_cells,
                boundary_sources,
                transition_cache;
                mode = mode
            )
        end
        for region_indices in components
            region_cells = affine_cells[region_indices]
            append!(grouped_cells, region_cells)
            push!(linear_regions, LinearRegion(Cell[Cell(cell) for cell in region_cells]))
        end
    end

    return grouped_cells, LinearRegions(linear_regions)
end
