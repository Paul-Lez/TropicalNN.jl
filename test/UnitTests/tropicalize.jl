using Test, TropicalNN, Oscar

@testset verbose = true "MLP to Tropical Conversion" begin
    #==========================================================================
    # Basic MLP Conversion Tests
    ==========================================================================#
    @testset verbose = true "tropicalize - Basic Conversion" begin
        # Test 1: Simple 2-layer network
        # Layer 1: 2 inputs -> 3 outputs (3x2 matrix)
        # Layer 2: 3 inputs -> 1 output (1x3 matrix)
        W = [Rational{BigInt}.([1 0; 0 1; -1 -1]), Rational{BigInt}.([1 1 1])]
        b = [Rational{BigInt}.([0, 0, 0]), Rational{BigInt}.([0])]
        t = [Rational{BigInt}.([0, 0, 0])]
        result = tropicalize(W, b, t)
        @test result isa Vector{<:RationalSignomial}
        @test length(result) == 1  # Single output
        deprecated_result = @test_deprecated mlp_to_trop(W, b, t)
        @test string(deprecated_result[1]) == string(result[1])
        result_default_thresholds = tropicalize(W, b)
        @test length(result_default_thresholds) == length(result)
        @test string(result_default_thresholds[1]) == string(result[1])

        # Test 4: Network with multiple outputs
        dims_multi = [2, 3, 2]
        W4, b4, t4 = random_mlp(dims_multi)
        result4 = tropicalize(W4, b4, t4)
        @test length(result4) == 2  # Two outputs
    end

    #==========================================================================
    # Error Handling Tests
    ==========================================================================#
    @testset verbose = true "Dimension Mismatch Errors" begin
        # Test 1: Mismatched bias dimensions
        W_bad = [Rational{BigInt}.([1 0; 0 1])]
        b_bad = [Rational{BigInt}.([0, 0, 0])]  # Wrong size (3 instead of 2)
        t_bad = Vector{Rational{BigInt}}[]
        @test_throws DimensionMismatch tropicalize(W_bad, b_bad, t_bad)

        # Test 2: Mismatched threshold dimensions
        W_bad2 = [Rational{BigInt}.([1 0; 0 1]), Rational{BigInt}.([1 1])]
        b_bad2 = [Rational{BigInt}.([0, 0]), Rational{BigInt}.([0])]
        t_bad2 = [Rational{BigInt}.([0])]  # Wrong size (1 instead of 2)
        @test_throws DimensionMismatch tropicalize(W_bad2, b_bad2, t_bad2)

        # Test 3: Layer dimension mismatch (second layer)
        W_layers = [Rational{BigInt}.([1 0; 0 1]), Rational{BigInt}.([1 1])]
        b_layers = [Rational{BigInt}.([0, 0]), Rational{BigInt}.([0, 0])]  # Wrong size
        t_layers = [Rational{BigInt}.([0, 0])]
        @test_throws DimensionMismatch tropicalize(W_layers, b_layers, t_layers)
    end

    #==========================================================================
    # single_to_trop Tests
    ==========================================================================#
    @testset verbose = true "single_to_trop - Single Layer Conversion" begin
        # Test 4: Dimension mismatch error
        A_bad = Rational{BigInt}.([1 0; 0 1])
        b_bad = Rational{BigInt}.([0, 0, 0])  # Wrong size
        t_bad = Rational{BigInt}.([0, 0])
        @test_throws DimensionMismatch single_to_trop(A_bad, b_bad, t_bad)
    end

    #==========================================================================
    # Composition Tests
    ==========================================================================#
    @testset verbose = true "Composition Operations" begin
        # Test 3: Correctness — comp(f, G) evaluated at p equals f evaluated at [G[i](p)]
        # f = max(x₁, x₂) as a polynomial in 2 variables
        # G = [1 + y₁, 2 + y₂] as rational signomials in 2 variables
        # comp(f, G) = max(1 + y₁, 2 + y₂)
        # At [y₁ = R(3), y₂ = R(5)]: max(1+3, 2+5) = max(4, 7) = R(7)
        # Use Rational{BigInt} exponents throughout to keep types consistent across `^`.
        R_comp = tropical_semiring(max)
        exps_comp = [Rational{BigInt}[1, 0], Rational{BigInt}[0, 1]]
        f_comp = Signomial([R_comp(0), R_comp(0)], exps_comp; sorted = false)
        g1 = signomial_to_rational(signomial_monomial(R_comp(1), Rational{BigInt}[1, 0]))
        g2 = signomial_to_rational(signomial_monomial(R_comp(2), Rational{BigInt}[0, 1]))
        composed_known = comp(f_comp, [g1, g2])
        composed_batched = comp(f_comp, [g1, g2]; quicksum = true)
        points_and_values = [
            ([R_comp(3), R_comp(5)], R_comp(7)),
            ([R_comp(10), R_comp(0)], R_comp(11)),
            ([R_comp(0), R_comp(0)], R_comp(2)),
            ([R_comp(5), R_comp(2)], R_comp(6)),
            ([R_comp(1), R_comp(1)], R_comp(3)),
            ([R_comp(-1), R_comp(5)], R_comp(7)),
            ([R_comp(100), R_comp(100)], R_comp(102))
        ]
        for (point, expected) in points_and_values
            @test TropicalNN.evaluate(composed_known, point) == expected
            @test TropicalNN.evaluate(composed_batched, point) == expected
        end

        composed_signomial = comp(f_comp, [g1.num, g2.num])
        composed_signomial_batched = comp(f_comp, [g1.num, g2.num]; quicksum = true)
        @test composed_signomial_batched == composed_signomial

        rational_outer = signomial_to_rational(f_comp)
        composed_rational = comp(rational_outer, [g1, g2]; quicksum = true)
        composed_vector = comp([rational_outer], [g1, g2]; quicksum = true)
        @test TropicalNN.evaluate(composed_rational, first(points_and_values)[1]) ==
              R_comp(7)
        @test TropicalNN.evaluate(only(composed_vector), first(points_and_values)[1]) ==
              R_comp(7)

        deprecated_composition = @test_deprecated TropicalNN.comp_with_quicksum(
            f_comp,
            [g1, g2]
        )
        @test TropicalNN.evaluate(deprecated_composition, first(points_and_values)[1]) ==
              R_comp(7)

        empty_outer = Signomial{Rational{BigInt}}(
            zeros(Rational{BigInt}, 2, 0),
            TropicalNN._TROPICAL_COEFF[],
            true
        )
        empty_sequential = comp(empty_outer, [g1, g2])
        empty_batched = comp(empty_outer, [g1, g2]; quicksum = true)
        @test empty_batched.num == empty_sequential.num
        @test empty_batched.den == empty_sequential.den

        # Test 5: Negative exponents require an explicit rational conversion.
        negative_outer = Signomial([R_comp(0)], [Rational{BigInt}[-1]]; sorted = false)
        signomial_input = signomial_monomial(R_comp(0), Rational{BigInt}[1])
        @test_throws ArgumentError comp(negative_outer, [signomial_input])

        rational_input = signomial_to_rational(signomial_input)
        explicit_composition = comp(negative_outer, [rational_input])
        @test TropicalNN.evaluate(explicit_composition, [R_comp(3)]) == R_comp(-3)
    end

    #==========================================================================
    # random_mlp Tests
    ==========================================================================#
    @testset verbose = true "random_mlp - Network Generation" begin
        # Test 1: Basic network generation with default parameters
        dims1 = [2, 3, 1]
        W1, b1, t1 = random_mlp(dims1)
        @test length(W1) == 2  # 2 layers
        @test length(b1) == 2
        @test length(t1) == 1
        @test size(W1[1]) == (3, 2)  # First layer: 3 neurons, 2 inputs
        @test size(W1[2]) == (1, 3)  # Second layer: 1 output, 3 inputs
        @test all(iszero, t1[1])  # Default thresholds are zero
        @test eltype(W1[1]) == Rational{BigInt}  # Default is symbolic

        # Test 2: Network with random thresholds
        W2, b2, t2 = random_mlp(dims1, random_thresholds = true)
        @test !all(iszero, t2[1])  # Thresholds should be non-zero

        # Test 3: Network with floating point (symbolic=false)
        W3, b3, t3 = random_mlp(dims1, symbolic = false)
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
        @test length(t5) == 0
        @test size(W5[1]) == (2, 3)
    end
end
