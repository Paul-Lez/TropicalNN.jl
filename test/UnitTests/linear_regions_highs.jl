using Test, TropicalNN, Oscar

@testset "Linear Regions HiGHS" begin
    R = tropical_semiring(max)

    # Test 1: Basic polynomial - max(x, y)
    @testset "Basic polynomial max(x, y)" begin
        u = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)

        # Test enum_linear_regions_highs (returns raw (A,b) feasibility pairs, not LinearRegions)
        regions_highs = enum_linear_regions_highs(u)
        @test length(regions_highs) == 2

        # Both regions should be feasible
        @test regions_highs[1][2] == true
        @test regions_highs[2][2] == true

        # Each region should be represented by (A, b) matrix pair
        @test regions_highs[1][1] isa Tuple{Matrix{Float64}, Vector{Float64}}
        @test regions_highs[2][1] isa Tuple{Matrix{Float64}, Vector{Float64}}
    end

    # Test 2: Rational function - max(x, y) / constant
    @testset "Rational function max(x, y) / 0" begin
        u = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        v = Signomial([R(0)], [[0//1, 0//1]]; sorted=false)
        q = u / v

        # enum_linear_regions_rat_highs now returns LinearRegions, mirroring the Oscar backend
        regions_rat_highs = enum_linear_regions_rat_highs(q)
        @test regions_rat_highs isa LinearRegions
        @test length(regions_rat_highs) == 2

        # Each LinearRegion contains (A, b) pairs
        for lr in regions_rat_highs
            @test lr isa LinearRegion
            @test length(lr.regions) >= 1
            @test lr.regions[1] isa Tuple{Matrix{Float64}, Vector{Float64}}
        end
    end

    # Test 3: More complex rational function
    @testset "Complex rational function" begin
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        g = Signomial([R(0), R(0)], [[1//1, 1//1], [1//1, 2//1]]; sorted=false)
        q = f / g

        regions_rat_highs = enum_linear_regions_rat_highs(q)
        @test regions_rat_highs isa LinearRegions
        @test length(regions_rat_highs) > 0
    end

    # Test 4: Polynomial with redundant monomial
    @testset "Polynomial max(0, x, 2x)" begin
        u = Signomial([R(0), R(0), R(0)], [[0//1], [1//1], [2//1]]; sorted=false)

        regions_highs = enum_linear_regions_highs(u)
        @test length(regions_highs) == 3

        feasible_count = sum([r[2] for r in regions_highs])
        @test feasible_count >= 2
    end

    # Test 5: Consistency check — both backends return the same region count
    @testset "HiGHS vs Oscar consistency" begin
        u = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        v = Signomial([R(0)], [[0//1, 0//1]]; sorted=false)
        q = u / v

        regions_oscar = enum_linear_regions_rat(q)
        regions_highs = enum_linear_regions_rat_highs(q)

        # Both return LinearRegions; counts should agree
        @test regions_oscar isa LinearRegions
        @test regions_highs isa LinearRegions
        @test length(regions_oscar) == length(regions_highs)
    end

    # Test 6: MLP-derived polynomial
    @testset "MLP-derived polynomial" begin
        W_fixed = [Rational{BigInt}.([1 0; 0 1]), Rational{BigInt}.([1 1])]
        b_fixed = [Rational{BigInt}.([0, 0]), Rational{BigInt}.([0])]
        t_fixed = [Rational{BigInt}.([0, 0]), Rational{BigInt}.([0])]
        trop_fixed = mlp_to_trop(W_fixed, b_fixed, t_fixed)[1]
        regions_highs_fixed = enum_linear_regions_rat_highs(trop_fixed)
        regions_oscar_fixed = enum_linear_regions_rat(trop_fixed)
        @test length(regions_highs_fixed) == length(regions_oscar_fixed)

        w, b, t = TropicalNN.random_mlp([2, 2, 1])
        trop = mlp_to_trop(w, b, t)[1]
        regions_highs = enum_linear_regions_rat_highs(trop)
        @test regions_highs isa LinearRegions
        @test length(regions_highs) > 0
    end

    # Test 7: Repeated linear map path (exists_reps = true)
    @testset "Repeated linear map (f/f) — 2D, 2 monomials" begin
        # f/f is the constant function 0; both diagonal pairs (i,i) share the same
        # linear map, so they should be collected into a single LinearRegion with 2 pieces.
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        lr = enum_linear_regions_rat_highs(f / f)
        @test lr isa LinearRegions
        @test length(lr) == 1          # one distinct linear map
        @test length(lr[1].regions) == 2  # two convex pieces
    end

    @testset "Repeated linear map (f/f) — 1D, 6 monomials" begin
        # f = max(0, x-1, 2x-4, 3x-9, 4x-16, 5x-25): 6 active monomials with breakpoints
        # at x = 1, 3, 5, 7, 9 → 6 full-dimensional regions.
        # f/f: all 6 diagonal pairs share linear map (coeff 0, exp [0]) → one LinearRegion
        # with 6 convex pieces.
        f6 = Signomial(
            [R(0), R(-1), R(-4), R(-9), R(-16), R(-25)],
            [[0//1], [1//1], [2//1], [3//1], [4//1], [5//1]];
            sorted=false
        )
        lr6 = enum_linear_regions_rat_highs(f6 / f6)
        @test lr6 isa LinearRegions
        @test length(lr6) == 1
        @test length(lr6[1].regions) == 6
        for (A, b) in lr6[1].regions
            @test TropicalNN.highs_is_full_dimensional(A, b)
        end
    end

    @testset "Repeated linear map (f/f) — 2D, 6 monomials" begin
        # f = max(0, y, 2y-1, x, x+y, x+2y-1): 2×3 grid of 6 full-dimensional regions.
        # f/f: all 6 diagonal pairs share linear map (coeff 0, exp [0,0]) → one LinearRegion
        # with 6 convex pieces.
        f6_2d = Signomial(
            [R(0), R(0), R(-1), R(0), R(0), R(-1)],
            [[0//1,0//1], [0//1,1//1], [0//1,2//1], [1//1,0//1], [1//1,1//1], [1//1,2//1]];
            sorted=false
        )
        lr6_2d = enum_linear_regions_rat_highs(f6_2d / f6_2d)
        @test lr6_2d isa LinearRegions
        @test length(lr6_2d) == 1
        @test length(lr6_2d[1].regions) == 6
        for (A, b) in lr6_2d[1].regions
            @test TropicalNN.highs_is_full_dimensional(A, b)
        end
    end

    # Test 8: Empty polyhedron detection
    @testset "Empty polyhedron detection" begin
        A = [1.0 0.0; -1.0 0.0]
        b = [0.0; -1.0]
        @test TropicalNN.highs_is_empty(A, b) == true

        A_feasible = [1.0 0.0; -1.0 0.0]
        b_feasible = [1.0; 0.0]
        @test TropicalNN.highs_is_empty(A_feasible, b_feasible) == false
    end

    # Test 9: Full dimensional check
    @testset "Full dimensional check" begin
        A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b = [1.0; 1.0; 1.0; 1.0]
        @test TropicalNN.highs_is_full_dimensional(A, b) == true

        A_line = [1.0 0.0; -1.0 0.0]
        b_line = [0.0; 0.0]
        @test TropicalNN.highs_is_full_dimensional(A_line, b_line) == false
    end
end
