using Test, TropicalNN, Oscar

@testset "Polynomial Algebra Operations" begin
    R = tropical_semiring(max)

    #==========================================================================
    # Addition Tests
    ==========================================================================#
    @testset "Polynomial Addition" begin
        # Test 1: Basic addition with Rational exponents
        f = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g = Signomial([R(3), R(4)], [[1//1, 0//1], [2//1, 0//1]], false)
        h = f + g
        @test length(h.exp) == 3  # Should have 3 unique monomials
        @test h.coeff[Rational{Int64}[1, 0]] == R(3)  # max(1, 3) = 3

        # Test 2: Addition with Float64 exponents (coeffs still Rational)
        f_flt = Signomial([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]], false)
        g_flt = Signomial([R(3), R(4)], [[1.0, 0.0], [2.0, 0.0]], false)
        h_flt = f_flt + g_flt
        @test length(h_flt.exp) == 3

        # Test 3: Addition with overlapping monomials
        f2 = Signomial([R(5), R(3)], [[1//1, 0//1], [0//1, 1//1]], false)
        g2 = Signomial([R(2), R(7)], [[1//1, 0//1], [0//1, 1//1]], false)
        h2 = f2 + g2
        @test length(h2.exp) == 2  # Same monomials, should merge
        @test h2.coeff[Rational{Int64}[1, 0]] == R(5)  # max(5, 2) = 5
        @test h2.coeff[Rational{Int64}[0, 1]] == R(7)  # max(3, 7) = 7

        # Test 4: Commutativity
        f4 = Signomial([R(1), R(2), R(3)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
        g4 = Signomial([R(4), R(5)], [[2//1, 0//1], [0//1, 2//1]], false)
        @test (f4 + g4).exp == (g4 + f4).exp
        @test (f4 + g4).coeff == (g4 + f4).coeff

        # Test 5: Associativity
        h_test = Signomial([R(6)], [[1//1, 2//1]], false)
        left = (f4 + g4) + h_test
        right = f4 + (g4 + h_test)
        @test Set(left.exp) == Set(right.exp)
        for exp_vec in left.exp
            @test left.coeff[exp_vec] == right.coeff[exp_vec]
        end
    end

    #==========================================================================
    # Multiplication Tests
    ==========================================================================#
    @testset "Polynomial Multiplication" begin
        # Test 1: Basic multiplication
        f = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g = Signomial([R(3), R(4)], [[1//1, 0//1], [0//1, 1//1]], false)
        h = f * g
        @test length(h.exp) <= 4  # At most 2*2 = 4 monomials
        # Check one specific monomial: (1,0) * (1,0) = (2,0) with coeff max(1+3) = 4
        @test haskey(h.coeff, Rational{Int64}[2, 0])
        @test h.coeff[Rational{Int64}[2, 0]] == R(4)

        # Test 2: Multiplication with Float64 exponents
        f_flt = Signomial([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]], false)
        g_flt = Signomial([R(3)], [[1.0, 0.0]], false)
        h_flt = f_flt * g_flt
        @test length(h_flt.exp) == 2

        # Test 3: Commutativity
        f3 = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g3 = Signomial([R(3), R(4), R(5)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
        left = f3 * g3
        right = g3 * f3
        @test Set(left.exp) == Set(right.exp)
        for exp_vec in left.exp
            @test left.coeff[exp_vec] == right.coeff[exp_vec]
        end

        # Test 4: Multiplication with quicksum variant
        f4 = Signomial([R(2), R(3)], [[1//1, 0//1], [0//1, 1//1]], false)
        g4 = Signomial([R(4), R(5)], [[1//1, 1//1], [2//1, 0//1]], false)
        h_std = f4 * g4
        h_qs = TropicalNN.mul_with_quicksum(f4, g4)
        # Results should be equivalent
        @test Set(h_std.exp) == Set(h_qs.exp)
        for exp_vec in h_std.exp
            @test h_std.coeff[exp_vec] == h_qs.coeff[exp_vec]
        end
    end

    #==========================================================================
    # Quicksum Tests
    ==========================================================================#
    @testset "Quicksum (Multi-polynomial Addition)" begin
        # Test 1: Basic quicksum with 3 polynomials
        polys = [
            Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false),
            Signomial([R(3), R(4)], [[1//1, 0//1], [2//1, 0//1]], false),
            Signomial([R(5)], [[0//1, 1//1]], false)
        ]
        result = TropicalNN.quicksum(polys)
        @test length(result.exp) >= 2  # At least 2 unique monomials

        # Test 2: Quicksum should give same result as sequential addition
        result_seq = polys[1] + polys[2] + polys[3]
        @test Set(result.exp) == Set(result_seq.exp)
        for exp_vec in result.exp
            @test result.coeff[exp_vec] == result_seq.coeff[exp_vec]
        end

        # Test 3: Quicksum with Float64 exponents
        polys_flt = [
            Signomial([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]], false),
            Signomial([R(3)], [[1.0, 0.0]], false),
            Signomial([R(4), R(5)], [[0.0, 1.0], [2.0, 0.0]], false)
        ]
        result_flt = TropicalNN.quicksum(polys_flt)
        @test length(result_flt.exp) >= 2

        # Test 4: Quicksum with many polynomials
        many_polys = [Signomial([R(i)], [[i//1, 0//1]], false) for i in 1:10]
        result_many = TropicalNN.quicksum(many_polys)
        @test length(result_many.exp) == 10  # All unique monomials

        # Test 5: Quicksum with single polynomial (edge case)
        single = [Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)]
        result_single = TropicalNN.quicksum(single)
        @test length(result_single.exp) == 2
    end

    #==========================================================================
    # Rational Function Tests
    ==========================================================================#
    @testset "Tropical Rational Functions" begin
        # Test 1: Basic rational function creation
        num = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        den = Signomial([R(0)], [[0//1, 0//1]], false)
        rat = RationalSignomial(num, den)
        @test rat.num == num
        @test rat.den == den

        # Test 2: Rational function addition
        num2 = Signomial([R(3)], [[1//1, 1//1]], false)
        den2 = Signomial([R(0)], [[0//1, 0//1]], false)
        rat2 = RationalSignomial(num2, den2)
        rat_sum = rat + rat2
        @test rat_sum isa RationalSignomial

        # Test 3: Rational function multiplication
        rat_prod = rat * rat2
        @test rat_prod isa RationalSignomial

        # Test 4: Rational function division
        rat_div = rat / rat2
        @test rat_div isa RationalSignomial
    end

    #==========================================================================
    # Evaluation Tests
    ==========================================================================#
    @testset "Polynomial Evaluation" begin
        # f = max(1 + x₁, 2 + x₂, 3 + x₁ + x₂)
        # at [R(2), R(3)]: max(1+2, 2+3, 3+2+3) = max(3, 5, 8) = 8
        f = Signomial([R(1), R(2), R(3)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
        @test TropicalNN.evaluate(f, [R(2), R(3)]) == R(8)

        # at [R(5), R(0)]: max(1+5, 2+0, 3+5+0) = max(6, 2, 8) = 8
        @test TropicalNN.evaluate(f, [R(5), R(0)]) == R(8)

        # at [R(0), R(10)]: max(1+0, 2+10, 3+0+10) = max(1, 12, 13) = 13
        @test TropicalNN.evaluate(f, [R(0), R(10)]) == R(13)

        # RationalSignomial evaluation: f/g where f = max(x₁, x₂), g = constant 0
        # at [R(2), R(3)]: num = max(0+2, 0+3) = R(3), den = R(0) → R(3)/R(0) = R(3)
        num = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]], false)
        den = Signomial([R(0)], [[0//1, 0//1]], false)
        q = RationalSignomial(num, den)
        @test TropicalNN.evaluate(q, [R(2), R(3)]) == R(3)
        # at [R(5), R(2)]: max(5, 2) = 5
        @test TropicalNN.evaluate(q, [R(5), R(2)]) == R(5)
    end

    #==========================================================================
    # Monomial Deduplication Tests
    ==========================================================================#
    @testset "Monomial Deduplication" begin
        # Test 1: Dedup with no zero-coefficient monomials (should be unchanged)
        g = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g_dedup = TropicalNN.dedup_monomials(g)
        @test length(g_dedup.exp) == length(g.exp)
        @test g_dedup.coeff == g.coeff

        # Test 2: Dedup actually removes a zero-coefficient monomial
        # Build a polynomial with one tropical-zero coefficient by summing f + (-f) for one monomial.
        # tropical zero = zero(R(0)) = R(-Inf)
        tropical_zero = zero(R(0))
        g_with_zero = Signomial([tropical_zero, R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g_with_zero_dedup = TropicalNN.dedup_monomials(g_with_zero)
        @test length(g_with_zero_dedup.exp) == 1
        @test g_with_zero_dedup.coeff[[0//1, 1//1]] == R(2)

        # Test 3: Dedup for rational functions
        num = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        den = Signomial([R(0)], [[0//1, 0//1]], false)
        rat = RationalSignomial(num, den)
        rat_dedup = TropicalNN.dedup_monomials(rat)
        @test rat_dedup.num isa Signomial
    end

    #==========================================================================
    # Edge Cases and Special Polynomials
    ==========================================================================#
    @testset "Edge Cases" begin
        # Test 1: Constant polynomial
        template = Signomial([R(1)], [[0//1, 0//1]], false)
        const_poly = Signomial_const(2, R(5), template)
        @test length(const_poly.exp) == 1
        @test const_poly.coeff[[0//1, 0//1]] == R(5)

        # Test 2: One polynomial (multiplicative identity in tropical arithmetic)
        one = Signomial_one(2, template)
        @test length(one.exp) == 1
        @test one.coeff[[0//1, 0//1]] == R(0)  # Tropical one is 0

        # Test 3: Single monomial
        mono = SignomialMonomial(R(3), [2//1, 1//1])
        @test length(mono.exp) == 1
        @test mono.coeff[[2//1, 1//1]] == R(3)

        # Test 4: Large number of variables
        large_exp = [i//1 for i in 1:20]
        f_large = Signomial([R(1)], [large_exp], false)
        @test length(f_large.exp[1]) == 20
    end

    #==========================================================================
    # Type Consistency Tests
    ==========================================================================#
    @testset "Type Consistency" begin
        # Test 1: Rational{Int64} operations maintain type
        f_r64 = Signomial([R(1), R(2)], [Rational{Int64}[1, 0], Rational{Int64}[0, 1]], false)
        g_r64 = Signomial([R(3)], [Rational{Int64}[1, 1]], false)
        h_r64 = f_r64 + g_r64
        @test eltype(h_r64.exp[1]) == Rational{Int64}

        # Test 2: Rational{BigInt} operations maintain type
        f_rbig = Signomial([R(1), R(2)], [Rational{BigInt}[1, 0], Rational{BigInt}[0, 1]], false)
        g_rbig = Signomial([R(3)], [Rational{BigInt}[1, 1]], false)
        h_rbig = f_rbig + g_rbig
        @test eltype(h_rbig.exp[1]) == Rational{BigInt}

        # Test 3: Float64 operations maintain type
        f_flt = Signomial([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]], false)
        g_flt = Signomial([R(3)], [[1.0, 1.0]], false)
        h_flt = f_flt + g_flt
        @test eltype(h_flt.exp[1]) == Float64
    end
end
