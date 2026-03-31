using Test, TropicalNN, Oscar

@testset "Main" begin
    w, b, t = TropicalNN.random_mlp([3, 2, 1])
    trop = mlp_to_trop(w, b, t)[1]
    @test length(enum_linear_regions_rat(trop)) > 0
    R = tropical_semiring(max)
    f = Signomial([R(1), R(2), R(3)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]]; sorted=false)
    # Write down the tropical polynomial 0*X^1*Y^7 + 4*X^0*Y^1 + (-5)*X^9*Y^1
    g = Signomial([R(0), R(4), R(-5)], [[1//1, 7//1], [0//1, 1//1], [9//1, 1//1]]; sorted=false) 

    # test addition and multiplication
    @test f + g == Signomial{Rational{Int64}}(Dict{Any, Any}(Rational{Int64}[1, 1] => R(3), Rational{Int64}[1, 0] => R(1), Rational{Int64}[1, 7] => R(0), Rational{Int64}[9, 1] => R(-5), Rational{Int64}[0, 1] => R(4)), Vector{Rational{Int64}}[[0, 1], [1, 0], [1, 1], [1, 7], [9, 1]])
    @test f * g == Signomial{Rational{Int64}}(Dict{Any, Any}(Rational{Int64}[2, 8] => R(3), Rational{Int64}[9, 2] => R(-3), Rational{Int64}[1, 8] => R(2), Rational{Int64}[10, 1] => R(-4), Rational{Int64}[0, 2] => R(6), Rational{Int64}[1, 1] => R(5), Rational{Int64}[10, 2] => R(-2), Rational{Int64}[2, 7] => R(1), Rational{Int64}[1, 2] => R(7)), Vector{Rational{Int64}}[[0, 2], [1, 1], [1, 2], [1, 8], [2, 7], [2, 8], [9, 2], [10, 1], [10, 2]])


    # test components
    V = [1, 2, 3, 4]
    D = Dict{Tuple{Int, Int}, Bool}((1, 2) => true, (3, 4) => true, (2, 3) => false)
    @test TropicalNN.components(V, D) == [[1, 2], [3, 4]]

    # test linear regions enumeration — no-repetitions path
    # Take u to be the tropical polynomial 0*X^1*Y^1 + 0*X^0*Y^1 = max(X, Y)
    u = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
    # and v to be the tropical polynomial 0
    v = Signomial([R(0)], [[0//1, 0//1]]; sorted=false)
    @test length(enum_linear_regions_rat(u / v)) == 2

    # test linear regions enumeration — repeated linear map (exists_reps = true)
    @testset "enum_linear_regions_rat repeated-map path via f/f" begin
        # f/f is the constant tropical function 0 (the multiplicative identity).
        # Every monomial pair (i, i) maps to the same linear map (coefficient 0, exponent 0⃗),
        # while cross-pairs (i, j) with i≠j intersect only on a lower-dimensional wall and
        # are discarded as not full-dimensional.
        #
        # f = max(x, y) has 2 regions: {x ≥ y} and {y ≥ x}. Both share the same linear
        # map, so they are collected into a single LinearRegion with 2 convex pieces.
        f = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        lr = enum_linear_regions_rat(f / f)

        @test lr isa LinearRegions
        # One distinct linear map → one LinearRegion
        @test length(lr) == 1
        # That LinearRegion contains both half-planes
        @test length(lr[1].regions) == 2
        # Every convex piece is full-dimensional
        @test all(Oscar.is_fulldimensional(p) for p in lr[1].regions)
    end

    # enum_linear_regions_rat repeated-map path — 6 monomials, 1 variable
    @testset "enum_linear_regions_rat repeated-map path via f/f — 6 monomials" begin
        # f = max(0, x-1, 2x-4, 3x-9, 4x-16, 5x-25).
        # The coefficients -c² lie on a concave curve, so all 6 monomials are active.
        # Breakpoints at x = 1, 3, 5, 7, 9 produce 6 full-dimensional regions.
        # f/f: all 6 diagonal pairs (i,i) share linear map (coefficient 0, exponent [0]).
        # → one LinearRegion containing 6 convex pieces.
        f6 = Signomial(
            [R(0), R(-1), R(-4), R(-9), R(-16), R(-25)],
            [[0//1], [1//1], [2//1], [3//1], [4//1], [5//1]];
            sorted=false
        )
        lr6 = enum_linear_regions_rat(f6 / f6)
        @test lr6 isa LinearRegions
        @test length(lr6) == 1
        @test length(lr6[1].regions) == 6
        @test all(Oscar.is_fulldimensional(p) for p in lr6[1].regions)
    end

    # enum_linear_regions_rat repeated-map path — 6 monomials, 2 variables
    @testset "enum_linear_regions_rat repeated-map path via f/f — 2D, 6 monomials" begin
        # f = max(0, y, 2y-1, x, x+y, x+2y-1)
        # This is the tropical product of max(0, x) and max(0, y, 2y-1), so its 6
        # linear regions are the 2×3 grid:
        #   {x≤0} × {y≤0}        {x≤0} × {0≤y≤1}      {x≤0} × {y≥1}
        #   {x≥0} × {y≤0}        {x≥0} × {0≤y≤1}      {x≥0} × {y≥1}
        # f/f: all 6 diagonal pairs share linear map (coefficient 0, exponent [0,0]).
        # → one LinearRegion containing 6 convex pieces.
        f6_2d = Signomial(
            [R(0), R(0), R(-1), R(0), R(0), R(-1)],
            [[0//1,0//1], [0//1,1//1], [0//1,2//1], [1//1,0//1], [1//1,1//1], [1//1,2//1]];
            sorted=false
        )
        lr6_2d = enum_linear_regions_rat(f6_2d / f6_2d)
        @test lr6_2d isa LinearRegions
        @test length(lr6_2d) == 1
        @test length(lr6_2d[1].regions) == 6
        @test all(Oscar.is_fulldimensional(p) for p in lr6_2d[1].regions)
    end

    # Monomial elimination
    # Take u to be the tropical polynomial max(0, x, 2x)
    u = Signomial([R(0), R(0), R(0)], [[0//1], [1//1], [2//1]]; sorted=false)
    # The monomial elimination of u should be max(0, 2x) since x is redundant
    @test monomial_strong_elim(u) == Signomial([R(0), R(0)], [[0//1], [2//1]]; sorted=false)

    # TODO: add tests for mlp_to_trop functions and the rest of the tropical algebra functions

end