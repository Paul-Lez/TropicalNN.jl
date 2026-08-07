using Test, TropicalNN, Oscar

struct UnsupportedLinearRegionsMode <: TropicalNN.LinearRegionsCalculationMode end

@testset verbose = true "Linear Regions General Calculation" begin
    R = tropical_semiring(max)
    oscar_mode = OscarMode()
    highs_mode = HiGHSMode()

    rational_region_signature(regions) = (
        length(regions), sort([length(region) for region in regions]))
    candidate_region(region) = only(region)
    candidate_is_feasible(candidate, mode) = TropicalNN.is_feasible(candidate_region(candidate); mode = mode)
    general_full_dimensional_flags(regions,
        mode) = [TropicalNN.is_full_dimensional(candidate_region(region); mode = mode)
                 for region in regions]

    @testset verbose = true "Polyhedron construction by mode" begin
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)

        oscar_region = TropicalNN.polyhedron(f, 1, oscar_mode)
        @test oscar_region isa Oscar.Polyhedron
        @test !(oscar_region isa Oscar.Polyhedron{Float64})

        exact_region = TropicalNN.make_polyhedron(
            reshape(Rational{BigInt}[1 // 1, -1 // 1], 2, 1),
            Rational{BigInt}[1 // 10, 1 // 3];
            mode = oscar_mode
        )
        exact_facets = Oscar.halfspace_matrix_pair(Oscar.facets(exact_region))
        exact_A = TropicalNN.get_matrix(exact_region; mode = oscar_mode)
        exact_b = TropicalNN.get_vector(exact_region; mode = oscar_mode)
        @test exact_A == exact_facets.A
        @test exact_b == exact_facets.b
        @test !(eltype(exact_A) <: AbstractFloat)
        @test !(eltype(exact_b) <: AbstractFloat)

        highs_region_1 = TropicalNN.polyhedron(f, 1, highs_mode)
        highs_region_2 = TropicalNN.polyhedron(f, 2, highs_mode)
        A1 = TropicalNN.get_matrix(highs_region_1; mode = highs_mode)
        b1 = TropicalNN.get_vector(highs_region_1; mode = highs_mode)
        A2 = TropicalNN.get_matrix(highs_region_2; mode = highs_mode)
        b2 = TropicalNN.get_vector(highs_region_2; mode = highs_mode)
        expected_A1 = permutedims(Vector{Float64}(TropicalNN.get_exp(f, 2)) -
                                  Vector{Float64}(TropicalNN.get_exp(f, 1)))
        expected_A2 = permutedims(Vector{Float64}(TropicalNN.get_exp(f, 1)) -
                                  Vector{Float64}(TropicalNN.get_exp(f, 2)))

        @test A1 == expected_A1
        @test b1 == [0.0]
        @test A2 == expected_A2
        @test b2 == [0.0]

        constant = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        for mode in (oscar_mode, highs_mode)
            A, b = TropicalNN._linear_region_constraint_data(constant, 1, mode)
            @test size(A) == (0, 2)
            @test isempty(b)
        end

        whole_space = TropicalNN.polyhedron(constant, 1, highs_mode)
        @test size(TropicalNN.get_matrix(whole_space; mode = highs_mode)) == (0, 2)
        @test TropicalNN.get_vector(whole_space; mode = highs_mode) == Float64[]
        @test TropicalNN.is_full_dimensional(whole_space; mode = highs_mode)
    end

    @testset verbose = true "Unsupported and empty inputs fail explicitly" begin
        @test_throws ArgumentError TropicalNN.make_polyhedron(
            zeros(Float64, 1, 1), Float64[0.0]; mode = UnsupportedLinearRegionsMode())

        empty = Signomial(Rational{BigInt}[], Vector{Vector{Rational{BigInt}}}(); sorted = false)
        @test_throws ArgumentError TropicalNN.linear_regions(
            RationalSignomial(empty, empty); mode = highs_mode)
    end

    @testset verbose = true "Polynomial region enumeration by mode" begin
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)

        oscar_regions = TropicalNN.linear_regions(f; mode = oscar_mode)

        @test length(oscar_regions) == 2
        @test all(region -> candidate_is_feasible(region, oscar_mode), oscar_regions)
        @test all(region -> candidate_region(region) isa Cell, oscar_regions)

        highs_regions = TropicalNN.linear_regions(f; mode = highs_mode)

        @test length(oscar_regions) == length(highs_regions)
        @test count(region -> candidate_is_feasible(region, oscar_mode), oscar_regions) ==
              count(region -> candidate_is_feasible(region, highs_mode), highs_regions)
        @test [candidate_is_feasible(region, highs_mode) for region in highs_regions] ==
              [true, true]
        @test all(region -> candidate_region(region) isa Cell, highs_regions)
        @test all(region -> !(candidate_region(region) isa TropicalNN._Polyhedra), highs_regions)
        @test all(
            region -> TropicalNN.get_matrix(candidate_region(region); mode = highs_mode) isa
                      Matrix{Float64},
            highs_regions)
        @test all(
            region -> TropicalNN.get_vector(candidate_region(region); mode = highs_mode) isa
                      Vector{Float64},
            highs_regions)

        for mode in (oscar_mode, highs_mode)
            partition = TropicalNN._signomial_region_partition([f]; mode = mode)
            coefficient_type = TropicalNN._linear_region_coefficient_type(mode)
            @test all(
                cell -> cell isa TropicalNN._Cell{Tuple{Int}, coefficient_type},
                partition
            )
            @test all(cell -> cell.A isa Matrix{coefficient_type}, partition)
            @test all(cell -> cell.b isa Vector{coefficient_type}, partition)
            @test sort([only(cell.data) for cell in partition]) == [1, 2]
        end
    end

    @testset verbose = true "Polynomial mode enumeration on edge cases" begin
        cases = [
            (
                "single monomial",
                Signomial([R(0)], [[0//1, 0//1]]; sorted = false),
                [1],
                [true]
            ),
            (
                "lower-dimensional dominance region",
                Signomial([R(0), R(0), R(0)], [[0//1], [1//1], [2//1]]; sorted = false),
                [1, 3],
                [true, true]
            ),
            (
                "empty dominance region",
                Signomial([R(0), R(0), R(-1)], [[0//1], [2//1], [1//1]]; sorted = false),
                [1, 3],
                [true, true]
            )
        ]

        for (label, f, expected_indices, expected_full_dimensional) in cases
            @testset "$label" begin
                general_oscar_regions = TropicalNN.linear_regions(f; mode = oscar_mode)
                general_highs_regions = TropicalNN.linear_regions(f; mode = highs_mode)

                expected_matrices = [permutedims(collect(TropicalNN.get_exp(f, i)))
                                     for i in expected_indices]
                @test [only(region).matrix for region in general_oscar_regions] ==
                      expected_matrices
                @test [only(region).matrix for region in general_highs_regions] ==
                      expected_matrices
                @test all(
                    region -> candidate_is_feasible(region, oscar_mode),
                    general_oscar_regions
                )
                @test all(
                    region -> candidate_is_feasible(region, highs_mode),
                    general_highs_regions
                )

                @test general_full_dimensional_flags(general_highs_regions, highs_mode) ==
                      expected_full_dimensional
                @test general_full_dimensional_flags(general_oscar_regions, oscar_mode) ==
                      expected_full_dimensional
            end
        end
    end

    @testset verbose = true "Vector enumeration filters infeasible candidates" begin
        has_empty_region = Signomial(
            [R(0), R(0), R(-1)],
            [[0//1], [2//1], [1//1]];
            sorted = false
        )
        constant = Signomial([R(0)], [[0//1]]; sorted = false)

        for mode in (oscar_mode, highs_mode)
            regions = TropicalNN.linear_regions(
                [has_empty_region, constant];
                mode = mode
            )
            expected_matrices = [
                reshape(Rational{BigInt}[0, 0], 2, 1),
                reshape(Rational{BigInt}[2, 0], 2, 1)
            ]
            @test [only(region).matrix for region in regions] == expected_matrices
            @test all(region -> candidate_is_feasible(region, mode), regions)
        end
    end

    @testset verbose = true "Rational mode enumeration" begin
        constant_1d = Signomial([R(0)], [[0//1]]; sorted = false)
        constant_2d = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        max_xy = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        lower_dimensional = Signomial(
            [R(0), R(0), R(0)],
            [[0//1], [1//1], [2//1]];
            sorted = false
        )
        has_empty_region = Signomial(
            [R(0), R(0), R(-1)],
            [[0//1], [2//1], [1//1]];
            sorted = false
        )
        six_adjacent_regions = Signomial(
            [R(0), R(-1), R(-4), R(-9), R(-16), R(-25)],
            [[0//1], [1//1], [2//1], [3//1], [4//1], [5//1]];
            sorted = false
        )

        cases = [
            ("single monomial quotient", constant_1d / constant_1d, (1, [1])),
            ("basic quotient", max_xy / constant_2d, (2, [1, 1])),
            ("lower-dimensional monomial discarded",
                lower_dimensional / constant_1d, (2, [1, 1])),
            ("empty monomial ignored", has_empty_region / constant_1d, (2, [1, 1])),
            ("boundary-glued repeated map", max_xy / max_xy, (1, [2])),
            ("many adjacent glued pieces",
                six_adjacent_regions / six_adjacent_regions, (1, [6]))
        ]

        for (label, q, expected_signature) in cases
            @testset "$label" begin
                general_oscar_regions = TropicalNN.linear_regions(q; mode = oscar_mode)
                general_highs_regions = TropicalNN.linear_regions(q; mode = highs_mode)

                @test rational_region_signature(general_oscar_regions) == expected_signature
                @test rational_region_signature(general_highs_regions) == expected_signature

                @test all(
                    TropicalNN.is_full_dimensional(piece; mode = oscar_mode)
                for
                region in general_oscar_regions for piece in region
                )
                @test all(
                    TropicalNN.is_full_dimensional(piece; mode = highs_mode)
                for
                region in general_highs_regions for piece in region
                )
            end
        end

        @testset "Cells store constraints and affine maps" begin
            regions = linear_regions(max_xy / constant_2d; mode = oscar_mode)
            cells = [cell for region in regions for cell in region]

            @test all(cell -> cell isa Cell, cells)
            @test first(regions).cells == collect(first(regions))
            @test all(cell -> size(cell.A, 1) == length(cell.b), cells)
            @test all(cell -> size(cell.matrix, 1) == length(cell.offset), cells)
            @test Set((Tuple(vec(cell.matrix)), Tuple(cell.offset)) for cell in cells) ==
                  Set([
                ((0 // 1, 1 // 1), (0 // 1,)),
                ((1 // 1, 0 // 1), (0 // 1,))
            ])
            @test fieldnames(typeof(first(cells))) == (:A, :b, :matrix, :offset)
        end
    end

    @testset "Disconnected repeated-map pieces match public HiGHS component splitting" begin
        f = Signomial([R(0), R(0), R(-2)], [[0//1], [1//1], [2//1]]; sorted = false)
        g = Signomial([R(0), R(-2)], [[0//1], [2//1]]; sorted = false)
        q = f / g

        general_oscar_regions = TropicalNN.linear_regions(q; mode = oscar_mode)
        general_highs_regions = TropicalNN.linear_regions(q; mode = highs_mode)

        @test rational_region_signature(general_highs_regions) == (4, [1, 1, 1, 1])
        @test rational_region_signature(general_oscar_regions) ==
              rational_region_signature(general_highs_regions)
    end
end
