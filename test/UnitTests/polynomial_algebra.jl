using Test, TropicalNN, Oscar

@testset verbose = true "Polynomial Algebra Operations" begin
    R = tropical_semiring(max)

    #==========================================================================
    # Addition Tests
    ==========================================================================#
    @testset verbose = true "Polynomial Addition" begin
        # Test 6: Constructor folds duplicate exponents with tropical addition
        duplicate = Signomial([R(3), R(1)], [[1//1, 0//1], [1//1, 0//1]]; sorted = false)
        @test length(duplicate) == 1
        @test TropicalNN.evaluate(duplicate, [R(0), R(0)]) == R(3)
    end

    #==========================================================================
    # Evaluation Tests
    ==========================================================================#
    @testset verbose = true "Polynomial Evaluation" begin
        f = Signomial(
            [R(1), R(2), R(3)], [
                [1//1, 0//1], [0//1, 1//1], [1//1, 1//1]]; sorted = false)
        @test TropicalNN.evaluate(f, [2, 3]) == R(8)
        @test TropicalNN.evaluate(f, [2.0, 3.0]) == R(8)
    end

    #==========================================================================
    # Edge Cases and Special Polynomials
    ==========================================================================#
    @testset verbose = true "Edge Cases" begin
        # Test 1: Constant polynomial
        template = Signomial([R(1)], [[0//1, 0//1]]; sorted = false)
        const_poly = signomial_const(2, R(5), template)
        @test length(const_poly) == 1
        @test TropicalNN.get_coeff_by_exp(const_poly, [0//1, 0//1]) == R(5)

        # Test 2: One polynomial (multiplicative identity in tropical arithmetic)
        one = signomial_one(2, template)
        @test length(one) == 1
        @test TropicalNN.get_coeff_by_exp(one, [0//1, 0//1]) == R(0)  # Tropical one is 0

        # Test 3: Single monomial
        mono = signomial_monomial(R(3), [2//1, 1//1])
        @test length(mono) == 1
        @test TropicalNN.get_coeff_by_exp(mono, [2//1, 1//1]) == R(3)

        # Test 4: Large number of variables
        large_exp = [i//1 for i in 1:20]
        f_large = Signomial([R(1)], [large_exp]; sorted = false)
        @test length(TropicalNN.get_exp(f_large, 1)) == 20

        # Test 5: Accessor arrays do not alias matrix-backed polynomial internals
        accessor_poly = Signomial(
            [R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        original_coeff = TropicalNN.get_coeff(accessor_poly, 1)
        coeffs = TropicalNN.coefficients(accessor_poly)
        coeffs[1] = R(99)
        @test TropicalNN.get_coeff(accessor_poly, 1) == original_coeff

        original_exp = TropicalNN.get_exp(accessor_poly, 1)
        exps = TropicalNN.exponents(accessor_poly)
        exps[1] = exps[2]
        @test TropicalNN.get_exp(accessor_poly, 1) == original_exp
    end

    @testset verbose = true "Deprecated factory names" begin
        template = Signomial([R(1)], [[0//1, 0//1]]; sorted = false)
        rational_template = signomial_to_rational(template)

        @test_deprecated TropicalNN.Signomial_const(2, R(5), template)
        @test_deprecated TropicalNN.Signomial_zero(2, template)
        @test_deprecated TropicalNN.Signomial_one(2, template)
        @test_deprecated TropicalNN.SignomialMonomial(R(3), [2//1, 1//1])
        @test_deprecated TropicalNN.RationalSignomial_identity(2, R(0))
        @test_deprecated TropicalNN.RationalSignomial_zero(2, rational_template)
        @test_deprecated TropicalNN.RationalSignomial_one(2, rational_template)
    end

    #==========================================================================
    # Type Consistency Tests
    ==========================================================================#
    @testset verbose = true "Type Consistency" begin
        # Test 1: Rational{Int64} operations maintain type
        f_r64 = Signomial(
            [R(1), R(2)], [Rational{Int64}[1, 0], Rational{Int64}[0, 1]]; sorted = false)
        g_r64 = Signomial([R(3)], [Rational{Int64}[1, 1]]; sorted = false)
        h_r64 = f_r64 + g_r64
        @test eltype(TropicalNN.exponents(h_r64)[1]) == Rational{Int64}

        # Test 2: Rational{BigInt} operations maintain type
        f_rbig = Signomial(
            [R(1), R(2)], [Rational{BigInt}[1, 0], Rational{BigInt}[0, 1]]; sorted = false)
        g_rbig = Signomial([R(3)], [Rational{BigInt}[1, 1]]; sorted = false)
        h_rbig = f_rbig + g_rbig
        @test eltype(TropicalNN.exponents(h_rbig)[1]) == Rational{BigInt}

        # Test 3: Float64 operations maintain type
        f_flt = Signomial([R(1), R(2)], [[1.0, 0.0], [0.0, 1.0]]; sorted = false)
        g_flt = Signomial([R(3)], [[1.0, 1.0]]; sorted = false)
        h_flt = f_flt + g_flt
        @test eltype(TropicalNN.exponents(h_flt)[1]) == Float64

        converted = convert(Signomial{Float64}, f_r64)
        @test TropicalNN.coefficients(converted) == TropicalNN.coefficients(f_r64)
        @test TropicalNN.exponents(converted) ==
              map(e -> Float64.(e), TropicalNN.exponents(f_r64))

        rational = signomial_to_rational(f_r64)
        converted_rational = convert(RationalSignomial{Float64}, rational)
        @test converted_rational.num == converted
    end
end
