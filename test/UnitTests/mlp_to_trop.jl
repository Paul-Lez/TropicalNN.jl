using Test, TropicalNN, Oscar

@testset "MLP to Tropical Conversion" begin
    R = tropical_semiring(max)

    #==========================================================================
    # Basic MLP Conversion Tests
    ==========================================================================#
    @testset "mlp_to_trop - Basic Conversion" begin
        # Test 1: Simple 2-layer network
        W = [Rational{BigInt}.([1 0; 0 1; -1 -1]), Rational{BigInt}.([1 1 1]')]
        b = [Rational{BigInt}.([0, 0, 0]), Rational{BigInt}.([0])]
        t = [Rational{BigInt}.([0, 0, 0]), Rational{BigInt}.([0])]
        result = mlp_to_trop(W, b, t)
        @test result isa Vector{TropicalPuiseuxRational{Rational{BigInt}}}
        @test length(result) == 1  # Single output

        # Test 2: Random small network with symbolic=true
        dims = [2, 3, 1]
        W2, b2, t2 = random_mlp(dims, symbolic=true)
        result2 = mlp_to_trop(W2, b2, t2)
        @test result2 isa Vector{TropicalPuiseuxRational{Rational{BigInt}}}
        @test length(result2) == 1

        # Test 3: Random small network with symbolic=false
        W3, b3, t3 = random_mlp(dims, symbolic=false)
        result3 = mlp_to_trop(W3, b3, t3)
        @test result3 isa Vector{TropicalPuiseuxRational{Float64}}
        @test length(result3) == 1

        # Test 4: Network with multiple outputs
        dims_multi = [2, 3, 2]
        W4, b4, t4 = random_mlp(dims_multi)
        result4 = mlp_to_trop(W4, b4, t4)
        @test length(result4) == 2  # Two outputs
    end

    #==========================================================================
    # Variant Function Tests
    ==========================================================================#
    @testset "mlp_to_trop Variants" begin
        dims = [2, 3, 1]
        W, b, t = random_mlp(dims)

        # Test 1: Standard version
        standard = mlp_to_trop(W, b, t)
        @test standard isa Vector{TropicalPuiseuxRational{Rational{BigInt}}}

        # Test 2: Quicksum version
        quicksum_result = mlp_to_trop_with_quicksum(W, b, t)
        @test quicksum_result isa Vector{TropicalPuiseuxRational{Rational{BigInt}}}

        # Test 3: Strong elimination version
        strong_elim = mlp_to_trop_with_strong_elim(W, b, t)
        @test strong_elim isa Vector{TropicalPuiseuxRational{Rational{BigInt}}}

        # Test 4: Quicksum with strong elimination
        qs_elim = mlp_to_trop_with_quicksum_with_strong_elim(W, b, t)
        @test qs_elim isa Vector{TropicalPuiseuxRational{Rational{BigInt}}}

        # Test 5: Dedup version
        dedup = mlp_to_trop_with_dedup(W, b, t)
        @test dedup isa Vector{TropicalPuiseuxRational{Rational{BigInt}}}

        # All variants should produce valid tropical rational functions
        for result in [standard, quicksum_result, strong_elim, qs_elim, dedup]
            @test length(result) == 1
            @test result[1] isa TropicalPuiseuxRational
        end
    end

    #==========================================================================
    # Error Handling Tests
    ==========================================================================#
    @testset "Dimension Mismatch Errors" begin
        # Test 1: Mismatched bias dimensions
        W_bad = [Rational{BigInt}.([1 0; 0 1])]
        b_bad = [Rational{BigInt}.([0, 0, 0])]  # Wrong size (3 instead of 2)
        t_bad = [Rational{BigInt}.([0, 0])]
        @test_throws DimensionMismatch mlp_to_trop(W_bad, b_bad, t_bad)

        # Test 2: Mismatched threshold dimensions
        W_bad2 = [Rational{BigInt}.([1 0; 0 1])]
        b_bad2 = [Rational{BigInt}.([0, 0])]
        t_bad2 = [Rational{BigInt}.([0])]  # Wrong size (1 instead of 2)
        @test_throws DimensionMismatch mlp_to_trop(W_bad2, b_bad2, t_bad2)

        # Test 3: Layer dimension mismatch (second layer)
        W_layers = [Rational{BigInt}.([1 0; 0 1]), Rational{BigInt}.([1 1]')]
        b_layers = [Rational{BigInt}.([0, 0]), Rational{BigInt}.([0, 0])]  # Wrong size
        t_layers = [Rational{BigInt}.([0, 0]), Rational{BigInt}.([0])]
        @test_throws DimensionMismatch mlp_to_trop(W_layers, b_layers, t_layers)
    end

    #==========================================================================
    # single_to_trop Tests
    ==========================================================================#
    @testset "single_to_trop - Single Layer Conversion" begin
        # Test 1: Simple identity-like layer
        A = Rational{BigInt}.([1 0; 0 1])
        b = Rational{BigInt}.([0, 0])
        t = Rational{BigInt}.([0, 0])
        result = single_to_trop(A, b, t)
        @test result isa Vector{TropicalPuiseuxRational{Rational{BigInt}}}
        @test length(result) == 2  # 2 outputs

        # Test 2: Layer with negative weights
        A2 = Rational{BigInt}.([1 -1; -1 1])
        b2 = Rational{BigInt}.([1, -1])
        t2 = Rational{BigInt}.([0, 0])
        result2 = single_to_trop(A2, b2, t2)
        @test length(result2) == 2

        # Test 3: Layer with non-zero thresholds
        A3 = Rational{BigInt}.([1 0; 0 1])
        b3 = Rational{BigInt}.([0, 0])
        t3 = Rational{BigInt}.([1, -1])  # Non-zero thresholds
        result3 = single_to_trop(A3, b3, t3)
        @test length(result3) == 2

        # Test 4: Dimension mismatch error
        A_bad = Rational{BigInt}.([1 0; 0 1])
        b_bad = Rational{BigInt}.([0, 0, 0])  # Wrong size
        t_bad = Rational{BigInt}.([0, 0])
        @test_throws DimensionMismatch single_to_trop(A_bad, b_bad, t_bad)
    end

    #==========================================================================
    # Composition Tests
    ==========================================================================#
    @testset "Composition Operations" begin
        # Create two simple layers
        W1, b1, t1 = random_mlp([2, 2])
        W2, b2, t2 = random_mlp([2, 2])

        layer1 = single_to_trop(W1[1], b1[1], t1[1])
        layer2 = single_to_trop(W2[1], b2[1], t2[1])

        # Test 1: Standard composition
        composed = comp(layer2, layer1)
        @test composed isa Vector{TropicalPuiseuxRational}
        @test length(composed) == 2

        # Test 2: Quicksum composition
        composed_qs = comp_with_quicksum(layer2, layer1)
        @test composed_qs isa Vector{TropicalPuiseuxRational}
        @test length(composed_qs) == 2

        # Both methods should produce valid results
        for c in [composed, composed_qs]
            @test all(x -> x isa TropicalPuiseuxRational, c)
        end
    end

    #==========================================================================
    # random_mlp Tests
    ==========================================================================#
    @testset "random_mlp - Network Generation" begin
        # Test 1: Basic network generation with default parameters
        dims1 = [2, 3, 1]
        W1, b1, t1 = random_mlp(dims1)
        @test length(W1) == 2  # 2 layers
        @test length(b1) == 2
        @test length(t1) == 2
        @test size(W1[1]) == (3, 2)  # First layer: 3 neurons, 2 inputs
        @test size(W1[2]) == (1, 3)  # Second layer: 1 output, 3 inputs
        @test all(iszero, t1[1])  # Default thresholds are zero
        @test eltype(W1[1]) == Rational{BigInt}  # Default is symbolic

        # Test 2: Network with random thresholds
        W2, b2, t2 = random_mlp(dims1, random_thresholds=true)
        @test !all(iszero, t2[1])  # Thresholds should be non-zero

        # Test 3: Network with floating point (symbolic=false)
        W3, b3, t3 = random_mlp(dims1, symbolic=false)
        @test eltype(W3[1]) == Float64
        @test eltype(b3[1]) == Float64

        # Test 4: Larger network
        dims_large = [5, 10, 8, 3]
        W4, b4, t4 = random_mlp(dims_large)
        @test length(W4) == 3  # 3 layers
        @test size(W4[1]) == (10, 5)
        @test size(W4[2]) == (8, 10)
        @test size(W4[3]) == (3, 8)

        # Test 5: Single layer network
        dims_single = [3, 2]
        W5, b5, t5 = random_mlp(dims_single)
        @test length(W5) == 1
        @test size(W5[1]) == (2, 3)
    end

    #==========================================================================
    # Edge Cases and Special Scenarios
    ==========================================================================#
    @testset "Edge Cases" begin
        # Test 1: Single input dimension
        dims_1d = [1, 2, 1]
        W, b, t = random_mlp(dims_1d)
        result = mlp_to_trop(W, b, t)
        @test result isa Vector{TropicalPuiseuxRational}
        @test length(result) == 1

        # Test 2: Wide hidden layer
        dims_wide = [2, 10, 1]
        W2, b2, t2 = random_mlp(dims_wide)
        result2 = mlp_to_trop(W2, b2, t2)
        @test result2 isa Vector{TropicalPuiseuxRational}

        # Test 3: Deep network
        dims_deep = [2, 3, 3, 3, 1]
        W3, b3, t3 = random_mlp(dims_deep)
        result3 = mlp_to_trop(W3, b3, t3)
        @test result3 isa Vector{TropicalPuiseuxRational}
        @test length(result3) == 1

        # Test 4: Network with all zero weights (degenerate case)
        W_zero = [zeros(Rational{BigInt}, 2, 2)]
        b_zero = [Rational{BigInt}.([1, 1])]
        t_zero = [Rational{BigInt}.([0, 0])]
        result_zero = mlp_to_trop(W_zero, b_zero, t_zero)
        @test result_zero isa Vector{TropicalPuiseuxRational}
    end

    #==========================================================================
    # Integration Tests - Combining Multiple Operations
    ==========================================================================#
    @testset "Integration Tests" begin
        # Test 1: Full pipeline with linear region enumeration
        dims = [2, 3, 1]
        W, b, t = random_mlp(dims)
        trop_func = mlp_to_trop(W, b, t)
        regions = enum_linear_regions_rat(trop_func[1])
        @test regions isa Vector
        @test length(regions) > 0

        # Test 2: Conversion and evaluation consistency
        W2, b2, t2 = random_mlp([2, 2, 1])
        trop_func2 = mlp_to_trop(W2, b2, t2)
        # Check that the result has proper structure for evaluation
        @test trop_func2[1] isa TropicalPuiseuxRational
        @test trop_func2[1].num isa TropicalPuiseuxPoly
        @test trop_func2[1].den isa TropicalPuiseuxPoly

        # Test 3: Monomial elimination actually reduces complexity
        W3, b3, t3 = random_mlp([2, 4, 1])
        without_elim = mlp_to_trop(W3, b3, t3)
        with_elim = mlp_to_trop_with_strong_elim(W3, b3, t3)
        # Both should produce valid results
        @test without_elim isa Vector{TropicalPuiseuxRational}
        @test with_elim isa Vector{TropicalPuiseuxRational}
        # Elimination version should have same or fewer monomials
        @test length(with_elim[1].num.exp) <= length(without_elim[1].num.exp)
    end

    #==========================================================================
    # Performance Characteristics Tests
    ==========================================================================#
    @testset "Performance Characteristics" begin
        # Test that quicksum variants complete without error
        # (actual performance testing would require BenchmarkTools)
        dims = [2, 4, 1]
        W, b, t = random_mlp(dims)

        # All variants should complete successfully
        @test mlp_to_trop(W, b, t) isa Vector{TropicalPuiseuxRational}
        @test mlp_to_trop_with_quicksum(W, b, t) isa Vector{TropicalPuiseuxRational}
        @test mlp_to_trop_with_strong_elim(W, b, t) isa Vector{TropicalPuiseuxRational}
        @test mlp_to_trop_with_quicksum_with_strong_elim(W, b, t) isa Vector{TropicalPuiseuxRational}
        @test mlp_to_trop_with_dedup(W, b, t) isa Vector{TropicalPuiseuxRational}
    end
end
