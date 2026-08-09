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

"""
    _Cell(A, b, matrix, offset, data)

Store an affine cell with data used by an internal computation. For a
signomial subdivision, `data` contains the dominant monomial indices. For a
rational subdivision, it contains the halfspace keys of the dominance cell.
The constraint matrix and vector have the same coefficient type.
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
