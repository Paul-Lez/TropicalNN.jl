using Test, TropicalNN, Oscar

@testset "Polynomial Algebra Operations" begin
    R = tropical_semiring(max)

    #==========================================================================
    # Addition Tests
    ==========================================================================#
    @testset "Polynomial Addition" begin
        # Test 1: Basic addition with Rational exponents
        f = TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g = TropicalPuiseuxPoly([R(3), R(4)], [[1//1, 0//1], [2//1, 0//1]], false)
        h = f + g
        @test length(h.exp) == 3  # Should have 3 unique monomials
        @test h.coeff[Rational{Int64}[1, 0]] == R(3)  # max(1, 3) = 3

        # Test 2: Addition with Float64 exponents (coeffs still Rational)
        f_flt = TropicalPuiseuxPoly([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]], false)
        g_flt = TropicalPuiseuxPoly([R(3), R(4)], [[1.0, 0.0], [2.0, 0.0]], false)
        h_flt = f_flt + g_flt
        @test length(h_flt.exp) == 3

        # Test 3: Addition with overlapping monomials
        f2 = TropicalPuiseuxPoly([R(5), R(3)], [[1//1, 0//1], [0//1, 1//1]], false)
        g2 = TropicalPuiseuxPoly([R(2), R(7)], [[1//1, 0//1], [0//1, 1//1]], false)
        h2 = f2 + g2
        @test length(h2.exp) == 2  # Same monomials, should merge
        @test h2.coeff[Rational{Int64}[1, 0]] == R(5)  # max(5, 2) = 5
        @test h2.coeff[Rational{Int64}[0, 1]] == R(7)  # max(3, 7) = 7

        # Test 4: Commutativity
        f4 = TropicalPuiseuxPoly([R(1), R(2), R(3)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
        g4 = TropicalPuiseuxPoly([R(4), R(5)], [[2//1, 0//1], [0//1, 2//1]], false)
        @test (f4 + g4).exp == (g4 + f4).exp
        @test (f4 + g4).coeff == (g4 + f4).coeff

        # Test 5: Associativity
        h_test = TropicalPuiseuxPoly([R(6)], [[1//1, 2//1]], false)
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
        f = TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g = TropicalPuiseuxPoly([R(3), R(4)], [[1//1, 0//1], [0//1, 1//1]], false)
        h = f * g
        @test length(h.exp) <= 4  # At most 2*2 = 4 monomials
        # Check one specific monomial: (1,0) * (1,0) = (2,0) with coeff max(1+3) = 4
        @test haskey(h.coeff, Rational{Int64}[2, 0])
        @test h.coeff[Rational{Int64}[2, 0]] == R(4)

        # Test 2: Multiplication with Float64 exponents
        f_flt = TropicalPuiseuxPoly([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]], false)
        g_flt = TropicalPuiseuxPoly([R(3)], [[1.0, 0.0]], false)
        h_flt = f_flt * g_flt
        @test length(h_flt.exp) == 2

        # Test 3: Commutativity
        f3 = TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g3 = TropicalPuiseuxPoly([R(3), R(4), R(5)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
        left = f3 * g3
        right = g3 * f3
        @test Set(left.exp) == Set(right.exp)
        for exp_vec in left.exp
            @test left.coeff[exp_vec] == right.coeff[exp_vec]
        end

        # Test 4: Multiplication with quicksum variant
        f4 = TropicalPuiseuxPoly([R(2), R(3)], [[1//1, 0//1], [0//1, 1//1]], false)
        g4 = TropicalPuiseuxPoly([R(4), R(5)], [[1//1, 1//1], [2//1, 0//1]], false)
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
            TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false),
            TropicalPuiseuxPoly([R(3), R(4)], [[1//1, 0//1], [2//1, 0//1]], false),
            TropicalPuiseuxPoly([R(5)], [[0//1, 1//1]], false)
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
            TropicalPuiseuxPoly([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]], false),
            TropicalPuiseuxPoly([R(3)], [[1.0, 0.0]], false),
            TropicalPuiseuxPoly([R(4), R(5)], [[0.0, 1.0], [2.0, 0.0]], false)
        ]
        result_flt = TropicalNN.quicksum(polys_flt)
        @test length(result_flt.exp) >= 2

        # Test 4: Quicksum with many polynomials
        many_polys = [TropicalPuiseuxPoly([R(i)], [[i//1, 0//1]], false) for i in 1:10]
        result_many = TropicalNN.quicksum(many_polys)
        @test length(result_many.exp) == 10  # All unique monomials

        # Test 5: Quicksum with single polynomial (edge case)
        single = [TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)]
        result_single = TropicalNN.quicksum(single)
        @test length(result_single.exp) == 2
    end

    #==========================================================================
    # Rational Function Tests
    ==========================================================================#
    @testset "Tropical Rational Functions" begin
        # Test 1: Basic rational function creation
        num = TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        den = TropicalPuiseuxPoly([R(0)], [[0//1, 0//1]], false)
        rat = TropicalPuiseuxRational(num, den)
        @test rat.num == num
        @test rat.den == den

        # Test 2: Rational function addition
        num2 = TropicalPuiseuxPoly([R(3)], [[1//1, 1//1]], false)
        den2 = TropicalPuiseuxPoly([R(0)], [[0//1, 0//1]], false)
        rat2 = TropicalPuiseuxRational(num2, den2)
        rat_sum = rat + rat2
        @test rat_sum isa TropicalPuiseuxRational

        # Test 3: Rational function multiplication
        rat_prod = rat * rat2
        @test rat_prod isa TropicalPuiseuxRational

        # Test 4: Rational function division
        rat_div = rat / rat2
        @test rat_div isa TropicalPuiseuxRational
    end

    #==========================================================================
    # Evaluation Tests
    ==========================================================================#
    @testset "Polynomial Evaluation" begin
        # Test 1: Basic evaluation with Rational
        f = TropicalPuiseuxPoly([R(1), R(2), R(3)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
        point = [R(2), R(3)]
        result = TropicalNN.eval(f, point)
        @test result isa Oscar.TropicalSemiringElem

        # Note: Float64 exponent evaluation not tested due to power operation issues
    end

    #==========================================================================
    # Monomial Deduplication Tests
    ==========================================================================#
    @testset "Monomial Deduplication" begin
        # Test 1: Dedup with no duplicates (should be unchanged)
        g = TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        g_dedup = TropicalNN.dedup_monomials(g)
        @test length(g_dedup.exp) == length(g.exp)
        @test g_dedup.coeff == g.coeff

        # Test 2: Dedup for rational functions
        num = TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)
        den = TropicalPuiseuxPoly([R(0)], [[0//1, 0//1]], false)
        rat = TropicalPuiseuxRational(num, den)
        rat_dedup = TropicalNN.dedup_monomials(rat)
        @test rat_dedup.num isa TropicalPuiseuxPoly
    end

    #==========================================================================
    # Edge Cases and Special Polynomials
    ==========================================================================#
    @testset "Edge Cases" begin
        # Test 1: Constant polynomial
        template = TropicalPuiseuxPoly([R(1)], [[0//1, 0//1]], false)
        const_poly = TropicalPuiseuxPoly_const(2, R(5), template)
        @test length(const_poly.exp) == 1
        @test const_poly.coeff[[0//1, 0//1]] == R(5)

        # Test 2: One polynomial (multiplicative identity in tropical arithmetic)
        one = TropicalPuiseuxPoly_one(2, template)
        @test length(one.exp) == 1
        @test one.coeff[[0//1, 0//1]] == R(0)  # Tropical one is 0

        # Test 3: Single monomial
        mono = TropicalPuiseuxMonomial(R(3), [2//1, 1//1])
        @test length(mono.exp) == 1
        @test mono.coeff[[2//1, 1//1]] == R(3)

        # Test 4: Large number of variables
        large_exp = [i//1 for i in 1:20]
        f_large = TropicalPuiseuxPoly([R(1)], [large_exp], false)
        @test length(f_large.exp[1]) == 20
    end

    #==========================================================================
    # Type Consistency Tests
    ==========================================================================#
    @testset "Type Consistency" begin
        # Test 1: Rational{Int64} operations maintain type
        f_r64 = TropicalPuiseuxPoly([R(1), R(2)], [Rational{Int64}[1, 0], Rational{Int64}[0, 1]], false)
        g_r64 = TropicalPuiseuxPoly([R(3)], [Rational{Int64}[1, 1]], false)
        h_r64 = f_r64 + g_r64
        @test eltype(h_r64.exp[1]) == Rational{Int64}

        # Test 2: Rational{BigInt} operations maintain type
        f_rbig = TropicalPuiseuxPoly([R(1), R(2)], [Rational{BigInt}[1, 0], Rational{BigInt}[0, 1]], false)
        g_rbig = TropicalPuiseuxPoly([R(3)], [Rational{BigInt}[1, 1]], false)
        h_rbig = f_rbig + g_rbig
        @test eltype(h_rbig.exp[1]) == Rational{BigInt}

        # Test 3: Float64 operations maintain type
        f_flt = TropicalPuiseuxPoly([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]], false)
        g_flt = TropicalPuiseuxPoly([R(3)], [[1.0, 1.0]], false)
        h_flt = f_flt + g_flt
        @test eltype(h_flt.exp[1]) == Float64
    end
end
