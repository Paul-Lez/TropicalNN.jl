using Test, TropicalNN, Oscar

@testset "Linear Regions HiGHS Mode" begin
    R = tropical_semiring(max)
    highs_mode = HiGHSMode()
    oscar_mode = OscarMode()

    @testset "Basic polynomial max(x, y)" begin
        u = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)

        regions_highs = enum_linear_regions_general(u; mode = highs_mode)
        @test length(regions_highs) == 2
        @test all(region -> region[2], regions_highs)
        @test all(
            region -> TropicalNN.get_matrix(region[1]; mode = highs_mode) isa Matrix{Float64},
            regions_highs
        )
        @test all(
            region -> TropicalNN.get_vector(region[1]; mode = highs_mode) isa Vector{Float64},
            regions_highs
        )
    end

    @testset "Rational function max(x, y) / 0" begin
        u = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        v = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        q = u / v

        regions_rat_highs = enum_linear_regions_rat_general(q; mode = highs_mode)
        @test regions_rat_highs isa LinearRegions
        @test length(regions_rat_highs) == 2

        for lr in regions_rat_highs
            @test lr isa LinearRegion
            @test length(lr.regions) >= 1
            @test TropicalNN.get_matrix(lr.regions[1]; mode = highs_mode) isa Matrix{Float64}
            @test TropicalNN.get_vector(lr.regions[1]; mode = highs_mode) isa Vector{Float64}
        end
    end

    @testset "Complex rational function" begin
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        g = Signomial([R(0), R(0)], [[1//1, 1//1], [1//1, 2//1]]; sorted = false)
        q = f / g

        regions_rat_highs = enum_linear_regions_rat_general(q; mode = highs_mode)
        @test regions_rat_highs isa LinearRegions
        @test length(regions_rat_highs) > 0
    end

    @testset "Polynomial max(0, x, 2x)" begin
        u = Signomial([R(0), R(0), R(0)], [[0//1], [1//1], [2//1]]; sorted = false)

        regions_highs = enum_linear_regions_general(u; mode = highs_mode)
        @test length(regions_highs) == 3

        feasible_count = sum([r[2] for r in regions_highs])
        @test feasible_count >= 2
    end

    @testset "HiGHS vs Oscar consistency" begin
        u = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        v = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        q = u / v

        regions_oscar = enum_linear_regions_rat_general(q; mode = oscar_mode)
        regions_highs = enum_linear_regions_rat_general(q; mode = highs_mode)

        @test regions_oscar isa LinearRegions
        @test regions_highs isa LinearRegions
        @test length(regions_oscar) == length(regions_highs)
    end

    @testset "MLP-derived polynomial" begin
        W_fixed = [Rational{BigInt}.([1 0; 0 1]), Rational{BigInt}.([1 1])]
        b_fixed = [Rational{BigInt}.([0, 0]), Rational{BigInt}.([0])]
        t_fixed = [Rational{BigInt}.([0, 0])]
        trop_fixed = mlp_to_trop(W_fixed, b_fixed, t_fixed)[1]
        regions_highs_fixed = enum_linear_regions_rat_general(trop_fixed; mode = highs_mode)
        regions_oscar_fixed = enum_linear_regions_rat_general(trop_fixed; mode = oscar_mode)
        @test length(regions_highs_fixed) == length(regions_oscar_fixed)

        w, b, t = TropicalNN.random_mlp([2, 2, 1])
        trop = mlp_to_trop(w, b, t)[1]
        regions_highs = enum_linear_regions_rat_general(trop; mode = highs_mode)
        @test regions_highs isa LinearRegions
        @test length(regions_highs) > 0
    end

    @testset "Repeated linear map (f/f) - 2D, 2 monomials" begin
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        lr = enum_linear_regions_rat_general(f / f; mode = highs_mode)
        @test lr isa LinearRegions
        @test length(lr) == 1
        @test length(lr[1].regions) == 2
    end

    @testset "Repeated linear map (f/f) - 1D, 6 monomials" begin
        f6 = Signomial(
            [R(0), R(-1), R(-4), R(-9), R(-16), R(-25)],
            [[0//1], [1//1], [2//1], [3//1], [4//1], [5//1]];
            sorted = false
        )
        lr6 = enum_linear_regions_rat_general(f6 / f6; mode = highs_mode)
        @test lr6 isa LinearRegions
        @test length(lr6) == 1
        @test length(lr6[1].regions) == 6
        for piece in lr6[1].regions
            @test TropicalNN.is_full_dimensional(piece; mode = highs_mode)
        end
    end

    @testset "Repeated linear map (f/f) - 2D, 6 monomials" begin
        f6_2d = Signomial(
            [R(0), R(0), R(-1), R(0), R(0), R(-1)],
            [[0//1, 0//1], [0//1, 1//1], [0//1, 2//1],
                [1//1, 0//1], [1//1, 1//1], [1//1, 2//1]];
            sorted = false
        )
        lr6_2d = enum_linear_regions_rat_general(f6_2d / f6_2d; mode = highs_mode)
        @test lr6_2d isa LinearRegions
        @test length(lr6_2d) == 1
        @test length(lr6_2d[1].regions) == 6
        for piece in lr6_2d[1].regions
            @test TropicalNN.is_full_dimensional(piece; mode = highs_mode)
        end
    end

    @testset "Empty polyhedron detection" begin
        A = [1.0 0.0; -1.0 0.0]
        b = [0.0; -1.0]
        @test TropicalNN.highs_is_empty(A, b) == true

        A_feasible = [1.0 0.0; -1.0 0.0]
        b_feasible = [1.0; 0.0]
        @test TropicalNN.highs_is_empty(A_feasible, b_feasible) == false
    end

    @testset "Dimension mismatch" begin
        @test_throws DimensionMismatch TropicalNN.highs_intersect_is_full_dimensional(
            zeros(Float64, 0, 1),
            Float64[],
            zeros(Float64, 0, 2),
            Float64[]
        )
    end

    @testset "Full dimensional check" begin
        A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b = [1.0; 1.0; 1.0; 1.0]
        @test TropicalNN.highs_is_full_dimensional(A, b) == true

        A_line = [1.0 0.0; -1.0 0.0]
        b_line = [0.0; 0.0]
        @test TropicalNN.highs_is_full_dimensional(A_line, b_line) == false

        A_infeasible_zero = zeros(Float64, 1, 1)
        b_infeasible_zero = [-1.0]
        @test TropicalNN.highs_is_empty(A_infeasible_zero, b_infeasible_zero) == true
        @test TropicalNN.highs_is_full_dimensional(
            A_infeasible_zero, b_infeasible_zero) == false

        A_tiny_line = [1e-7; -1e-7;;]
        b_tiny_line = [0.0; 0.0]
        @test TropicalNN.highs_is_full_dimensional(A_tiny_line, b_tiny_line) == false
    end
end
