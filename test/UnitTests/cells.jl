using Test, TropicalNN

@testset verbose = true "Affine cells" begin
    A = [1.0 0.0; 2.0 0.0; -1.0 0.0; 0.0 0.0]
    b = [1.0, 2.0, -1.0, 0.0]
    matrix = reshape(Rational{BigInt}[2, 3], 1, 2)
    offset = Rational{BigInt}[4]

    cell = TropicalNN.Cell(A, b, matrix, offset)
    internal_cell = TropicalNN._Cell(A, b, matrix, offset, nothing)

    @test cell.A === A
    @test cell.b === b
    @test cell.matrix === matrix
    @test cell.offset === offset
    @test internal_cell isa TropicalNN._Cell{Nothing, Float64}
    @test internal_cell.data === nothing
    @test_throws MethodError TropicalNN._Cell(
        A,
        Rational{BigInt}.(b),
        matrix,
        offset,
        nothing
    )

    public_cell = TropicalNN.Cell(internal_cell)
    @test public_cell.A === internal_cell.A
    @test public_cell.b === internal_cell.b
    @test public_cell.matrix === internal_cell.matrix
    @test public_cell.offset === internal_cell.offset

    full_dimensional_cell = TropicalNN._Cell(
        [1.0 0.0; -1.0 0.0],
        [1.0, 1.0],
        matrix,
        offset,
        (1,)
    )
    @test full_dimensional_cell isa TropicalNN._Cell{Tuple{Int}}
    for mode in (OscarMode(), HiGHSMode())
        @test TropicalNN.get_matrix(internal_cell; mode = mode) === A
        @test TropicalNN.get_vector(internal_cell; mode = mode) === b
        @test TropicalNN.is_feasible(internal_cell; mode = mode)
        @test !TropicalNN.is_full_dimensional(internal_cell; mode = mode)
        @test TropicalNN.is_full_dimensional(full_dimensional_cell; mode = mode)
    end

    key = [
        (1 // 1, Rational{BigInt}[2, 3]),
        (-1 // 1, Rational{BigInt}[0, 4])
    ]
    affine_matrix, affine_offset = TropicalNN._affine_formula_from_linear_map_key(key)
    @test affine_matrix == Rational{BigInt}[2 3; 0 4]
    @test affine_offset == Rational{BigInt}[1, -1]
    @test TropicalNN._affine_map_key(affine_matrix, affine_offset) ==
          TropicalNN._affine_map_key(copy(affine_matrix), copy(affine_offset))
end
