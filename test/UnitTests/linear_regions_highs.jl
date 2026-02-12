using Test, TropicalNN, Oscar

@testset "Linear Regions HiGHS" begin
    R = tropical_semiring(max)

    # Test 1: Basic polynomial - max(x, y)
    @testset "Basic polynomial max(x, y)" begin
        u = TropicalPuiseuxPoly([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]], false)

        # Test enum_linear_regions_highs
        regions_highs = enum_linear_regions_highs(u)
        @test length(regions_highs) == 2

        # Both regions should be feasible
        @test regions_highs[1][2] == true  # First region is feasible
        @test regions_highs[2][2] == true  # Second region is feasible

        # Each region should be represented by (A, b) matrix pair
        @test regions_highs[1][1] isa Tuple{Matrix{Float64}, Vector{Float64}}
        @test regions_highs[2][1] isa Tuple{Matrix{Float64}, Vector{Float64}}
    end

    # Test 2: Rational function - max(x, y) / constant
    @testset "Rational function max(x, y) / 0" begin
        u = TropicalPuiseuxPoly([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]], false)
        v = TropicalPuiseuxPoly([R(0)], [[0//1, 0//1]], false)
        q = u / v

        # Test enum_linear_regions_rat_highs
        regions_rat_highs = enum_linear_regions_rat_highs(q)
        @test length(regions_rat_highs) == 2

        # Each region should be (A, b) pair
        for region in regions_rat_highs
            @test region isa Tuple{Matrix{Float64}, Vector{Float64}}
        end
    end

    # Test 3: More complex rational function
    @testset "Complex rational function" begin
        # f = max(x, y) and g = max(x+y, x+2y)
        f = TropicalPuiseuxPoly([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]], false)
        g = TropicalPuiseuxPoly([R(0), R(0)], [[1//1, 1//1], [1//1, 2//1]], false)
        q = f / g

        regions_rat_highs = enum_linear_regions_rat_highs(q)
        # Should have some regions (exact count depends on geometry)
        @test length(regions_rat_highs) > 0
    end

    # Test 4: Polynomial with redundant monomial
    @testset "Polynomial max(0, x, 2x)" begin
        # max(0, x, 2x) - the x monomial is redundant
        u = TropicalPuiseuxPoly([R(0), R(0), R(0)], [[0//1], [1//1], [2//1]], false)

        regions_highs = enum_linear_regions_highs(u)
        @test length(regions_highs) == 3  # Still gets all three before elimination

        # Check that at least some regions are feasible
        feasible_count = sum([r[2] for r in regions_highs])
        @test feasible_count >= 2  # At least 0 and 2x should be feasible
    end

    # Test 5: Consistency check - compare HiGHS with Oscar on simple case
    @testset "HiGHS vs Oscar consistency" begin
        u = TropicalPuiseuxPoly([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]], false)
        v = TropicalPuiseuxPoly([R(0)], [[0//1, 0//1]], false)
        q = u / v

        # Get regions from both implementations
        regions_oscar = enum_linear_regions_rat(q)
        regions_highs = enum_linear_regions_rat_highs(q)

        # Should have the same number of regions
        @test length(regions_oscar) == length(regions_highs)
    end

    # Test 6: MLP-derived polynomial
    @testset "MLP-derived polynomial" begin
        # Create a small random MLP and convert to tropical
        w, b, t = TropicalNN.random_mlp([2, 2, 1])
        trop = mlp_to_trop(w, b, t)[1]

        # Test that enum_linear_regions_rat_highs works
        regions_highs = enum_linear_regions_rat_highs(trop)
        @test length(regions_highs) > 0

        # Compare with Oscar version
        regions_oscar = enum_linear_regions_rat(trop)
        @test length(regions_highs) == length(regions_oscar)
    end

    # Test 7: Empty polyhedron detection
    @testset "Empty polyhedron detection" begin
        # Create a deliberately infeasible system: x ≤ 0 and x ≥ 1
        A = [1.0 0.0; -1.0 0.0]  # 2x2 matrix
        b = [0.0; -1.0]

        # This should be detected as empty
        @test TropicalNN.highs_is_empty(A, b) == true

        # Feasible system: x ≤ 1 and x ≥ 0
        A_feasible = [1.0 0.0; -1.0 0.0]  # 2x2 matrix
        b_feasible = [1.0; 0.0]
        @test TropicalNN.highs_is_empty(A_feasible, b_feasible) == false
    end

    # Test 8: Full dimensional check
    @testset "Full dimensional check" begin
        # 2D box: x ≤ 1, x ≥ -1, y ≤ 1, y ≥ -1 (full dimensional)
        A = [1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0]
        b = [1.0; 1.0; 1.0; 1.0]
        @test TropicalNN.highs_is_full_dimensional(A, b) == true

        # 1D line in 2D: x = 0 (not full dimensional in 2D)
        A_line = [1.0 0.0; -1.0 0.0]
        b_line = [0.0; 0.0]
        @test TropicalNN.highs_is_full_dimensional(A_line, b_line) == false
    end
end
