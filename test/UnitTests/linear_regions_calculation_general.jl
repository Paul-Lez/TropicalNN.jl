using Test, TropicalNN, Oscar

struct UnsupportedLinearRegionsMode <: TropicalNN.LinearRegionsCalculationMode end

@testset "Linear Regions General Calculation" begin
    R = tropical_semiring(max)
    oscar_mode = OscarMode()
    highs_mode = HiGHSMode()

    rational_region_signature(regions) = (
        length(regions), sort([length(region) for region in regions]))
    rational_piece_count(regions) = sum([length(region) for region in regions])
    general_full_dimensional_flags(regions, mode) = [TropicalNN.is_full_dimensional(region[1]; mode = mode)
                                                     for region in regions]

    @testset "Polyhedron construction by mode" begin
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)

        oscar_region = TropicalNN.polyhedron(f, 1, oscar_mode)
        @test oscar_region isa Oscar.Polyhedron
        @test !(oscar_region isa Oscar.Polyhedron{Float64})

        highs_region_1 = TropicalNN.polyhedron(f, 1, highs_mode)
        highs_region_2 = TropicalNN.polyhedron(f, 2, highs_mode)
        A1 = TropicalNN.get_matrix(highs_region_1; mode = highs_mode)
        b1 = TropicalNN.get_vector(highs_region_1; mode = highs_mode)
        A2 = TropicalNN.get_matrix(highs_region_2; mode = highs_mode)
        b2 = TropicalNN.get_vector(highs_region_2; mode = highs_mode)
        expected_A1 = permutedims(Vector{Float64}(get_exp(f, 2)) -
                                  Vector{Float64}(get_exp(f, 1)))
        expected_A2 = permutedims(Vector{Float64}(get_exp(f, 1)) -
                                  Vector{Float64}(get_exp(f, 2)))

        @test A1 == expected_A1
        @test b1 == [0.0]
        @test A2 == expected_A2
        @test b2 == [0.0]

        constant = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        whole_space = TropicalNN.polyhedron(constant, 1, highs_mode)
        @test size(TropicalNN.get_matrix(whole_space; mode = highs_mode)) == (0, 2)
        @test TropicalNN.get_vector(whole_space; mode = highs_mode) == Float64[]
        @test TropicalNN.is_full_dimensional(whole_space; mode = highs_mode)
    end

    @testset "Unsupported and empty inputs fail explicitly" begin
        @test_throws ArgumentError TropicalNN.make_polyhedron(
            zeros(Float64, 1, 1), Float64[0.0]; mode = UnsupportedLinearRegionsMode())

        empty = Signomial(Rational{BigInt}[], Vector{Vector{Rational{BigInt}}}(); sorted = false)
        @test_throws ArgumentError TropicalNN.enum_linear_regions_rat_general(
            RationalSignomial(empty, empty); mode = highs_mode)
    end

    @testset "Polynomial region enumeration by mode" begin
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)

        oscar_regions = TropicalNN.enum_linear_regions_general(f; mode = oscar_mode)

        @test length(oscar_regions) == 2
        @test all(region -> region[2], oscar_regions)
        @test all(region -> region[1] isa Oscar.Polyhedron, oscar_regions)

        highs_regions = TropicalNN.enum_linear_regions_general(f; mode = highs_mode)

        @test length(oscar_regions) == length(highs_regions)
        @test count(region -> region[2], oscar_regions) ==
              count(region -> region[2], highs_regions)
        @test [region[2] for region in highs_regions] == [true, true]
        @test all(
            region -> TropicalNN.get_matrix(region[1]; mode = highs_mode) isa
                      Matrix{Float64},
            highs_regions)
        @test all(
            region -> TropicalNN.get_vector(region[1]; mode = highs_mode) isa
                      Vector{Float64},
            highs_regions)
    end

    @testset "Polynomial mode enumeration on edge cases" begin
        cases = [
            (
                "single monomial",
                Signomial([R(0)], [[0//1, 0//1]]; sorted = false),
                [true],
                [true]
            ),
            (
                "lower-dimensional dominance region",
                Signomial([R(0), R(0), R(0)], [[0//1], [1//1], [2//1]]; sorted = false),
                [true, true, true],
                [true, false, true]
            ),
            (
                "empty dominance region",
                Signomial([R(0), R(0), R(-1)], [[0//1], [2//1], [1//1]]; sorted = false),
                [true, false, true],
                [true, false, true]
            )
        ]

        for (label, f, expected_feasible, expected_full_dimensional) in cases
            @testset "$label" begin
                general_oscar_regions = TropicalNN.enum_linear_regions_general(f; mode = oscar_mode)
                general_highs_regions = TropicalNN.enum_linear_regions_general(f; mode = highs_mode)

                @test [region[2] for region in general_oscar_regions] == expected_feasible
                @test [region[2] for region in general_highs_regions] == expected_feasible

                @test general_full_dimensional_flags(general_highs_regions, highs_mode) ==
                      expected_full_dimensional
                @test general_full_dimensional_flags(general_oscar_regions, oscar_mode) ==
                      expected_full_dimensional
            end
        end
    end

    @testset "Rational mode enumeration" begin
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
                general_oscar_regions = TropicalNN.enum_linear_regions_rat_general(q; mode = oscar_mode)
                general_highs_regions = TropicalNN.enum_linear_regions_rat_general(q; mode = highs_mode)

                @test rational_region_signature(general_oscar_regions) == expected_signature
                @test rational_region_signature(general_highs_regions) == expected_signature
                @test rational_region_signature(general_oscar_regions) ==
                      rational_region_signature(general_highs_regions)

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
    end

    @testset "Disconnected repeated-map pieces match public HiGHS component splitting" begin
        f = Signomial([R(0), R(0), R(-2)], [[0//1], [1//1], [2//1]]; sorted = false)
        g = Signomial([R(0), R(-2)], [[0//1], [2//1]]; sorted = false)
        q = f / g

        general_oscar_regions = TropicalNN.enum_linear_regions_rat_general(q; mode = oscar_mode)
        general_highs_regions = TropicalNN.enum_linear_regions_rat_general(q; mode = highs_mode)

        @test rational_region_signature(general_highs_regions) == (4, [1, 1, 1, 1])
        @test rational_region_signature(general_oscar_regions) ==
              rational_region_signature(general_highs_regions)
        @test rational_piece_count(general_oscar_regions) ==
              rational_piece_count(general_highs_regions)
    end

    @testset "HiGHS rational region enumeration" begin
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        g = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        q = f / g

        highs_regions = TropicalNN.enum_linear_regions_rat_general(q; mode = highs_mode)

        @test highs_regions isa LinearRegions
        @test length(highs_regions) == 2
        @test sort([length(region) for region in highs_regions]) == [1, 1]
        @test all(
            TropicalNN.is_full_dimensional(piece; mode = highs_mode)
        for
        region in highs_regions for piece in region
        )
    end

    @testset "Repeated linear map components" begin
        f = Signomial(
            [R(0), R(-1), R(-4), R(-9), R(-16), R(-25)],
            [[0//1], [1//1], [2//1], [3//1], [4//1], [5//1]];
            sorted = false
        )

        regions = TropicalNN.enum_linear_regions_rat_general(f / f; mode = highs_mode)
        @test regions isa LinearRegions
        @test length(regions) == 1
        @test length(regions[1]) == 6
        @test all(
            TropicalNN.is_full_dimensional(piece; mode = highs_mode) for region in regions
        for piece in region
        )
    end

    @testset "Disconnected pieces with the same linear map are split" begin
        f = Signomial([R(0), R(0), R(-2)], [[0//1], [1//1], [2//1]]; sorted = false)
        g = Signomial([R(0), R(-2)], [[0//1], [2//1]]; sorted = false)
        q = f / g

        regions = TropicalNN.enum_linear_regions_rat_general(q; mode = highs_mode)
        @test regions isa LinearRegions
        @test length(regions) == 4
        @test sort([length(region) for region in regions]) == [1, 1, 1, 1]
        @test all(
            TropicalNN.is_full_dimensional(piece; mode = highs_mode) for region in regions
        for piece in region
        )
    end
end
