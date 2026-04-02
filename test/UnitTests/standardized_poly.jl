using Test, TropicalNN, Oscar

@testset "Standardized Polynomial Conversion" begin
    R = tropical_semiring(max)

    #==========================================================================
    # Basic Conversion Tests
    ==========================================================================#
    @testset "standardize() and destandardize()" begin
        # Test 1: Simple polynomial with rational exponents
        f = Signomial([R(1), R(2)], [[1//2, 0//1], [0//1, 1//3]]; sorted=false)
        g = TropicalNN.standardize(f)

        @test g isa TropicalNN.StandardizedTropicalPoly
        @test g.denominator == 6  # lcm(2, 3) = 6
        @test nvars(g.ring) == 2

        # Test 2: Round-trip conversion
        f_back = TropicalNN.destandardize(g)
        @test f_back isa AbstractSignomial
        @test Set(f_back.exp) == Set(f.exp)
        @test f_back.coeff == f.coeff

        # Test 3: Polynomial with integer exponents (denominator should be 1)
        f_int = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 2//1]]; sorted=false)
        g_int = TropicalNN.standardize(f_int)
        @test g_int.denominator == 1

        # Test 4: Complex denominators
        f_complex = Signomial(
            [R(1), R(2), R(3)],
            [[1//2, 3//4], [2//3, 1//6], [1//12, 5//8]],
            false
        )
        g_complex = TropicalNN.standardize(f_complex)
        # lcm(2, 4, 3, 6, 12, 8) = 24
        @test g_complex.denominator == 24

        # Test round-trip preserves polynomial
        f_complex_back = TropicalNN.destandardize(g_complex)
        @test Set(f_complex_back.exp) == Set(f_complex.exp)
        @test f_complex_back.coeff == f_complex.coeff
    end

    #==========================================================================
    # Denominator Conversion Tests
    ==========================================================================#
    @testset "convert_denominator()" begin
        # Create standardized polynomial with denominator 2
        f = Signomial([R(1), R(2)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g = TropicalNN.standardize(f)
        @test g.denominator == 2

        # Convert to denominator 6
        h = TropicalNN.convert_denominator(g, 6)
        @test h.denominator == 6
        @test nvars(h.ring) == nvars(g.ring)

        # Converting back to original denominator via destandardize should preserve values
        g_back = TropicalNN.destandardize(h)
        f_back = TropicalNN.destandardize(g)
        @test Set(g_back.exp) == Set(f_back.exp)
        @test g_back.coeff == f_back.coeff
    end

    #==========================================================================
    # Constructor Tests
    ==========================================================================#
    @testset "Standardized Polynomial Constructors" begin
        # Test 1: Constant polynomial
        const_poly = TropicalNN.StandardizedTropicalPoly_const(2, R(5), 1)
        @test const_poly.denominator == 1
        @test nvars(const_poly.ring) == 2
        # Evaluate at origin should give constant
        @test TropicalNN.evaluate(const_poly, [R(0), R(0)]) == R(5)

        # Test 2: Zero polynomial
        zero_poly = TropicalNN.StandardizedTropicalPoly_zero(3, 1, R)
        @test TropicalNN.evaluate(zero_poly, [R(1), R(1), R(1)]) == zero(R(0))

        # Test 3: One polynomial (multiplicative identity)
        one_poly = TropicalNN.StandardizedTropicalPoly_one(2, 1, R)
        @test TropicalNN.evaluate(one_poly, [R(1), R(1)]) == R(0)  # Tropical one is 0

        # Test 4: Single monomial
        mono = TropicalNN.StandardizedTropicalMonomial(R(3), [2, 1], 1)
        @test mono.denominator == 1
        # Evaluate at (1, 1) should give 3 + 2*1 + 1*1 = 6 in tropical arithmetic
        @test TropicalNN.evaluate(mono, [R(1), R(1)]) == R(3) * R(1)^2 * R(1)
    end

    #==========================================================================
    # Edge Cases
    ==========================================================================#
    @testset "Edge Cases" begin
        # Test 1: Single term polynomial
        f_single = Signomial([R(5)], [[2//3, 1//2]]; sorted=false)
        g_single = TropicalNN.standardize(f_single)
        @test g_single.denominator == 6
        # Round-trip should preserve values
        f_single_back = TropicalNN.destandardize(g_single)
        @test Set(f_single_back.exp) == Set(f_single.exp)
        @test f_single_back.coeff == f_single.coeff

        # Test 2: Polynomial with many variables
        large_exp = [i//2 for i in 1:10]
        f_large = Signomial([R(1)], [large_exp]; sorted=false)
        g_large = TropicalNN.standardize(f_large)
        @test g_large.denominator == 2
        @test nvars(g_large.ring) == 10

        # Test 3: Already integer exponents
        f_already_int = Signomial([R(1), R(2)], [[2//1, 3//1], [1//1, 5//1]]; sorted=false)
        g_already_int = TropicalNN.standardize(f_already_int)
        @test g_already_int.denominator == 1
        # Round-trip should preserve values
        f_already_int_back = TropicalNN.destandardize(g_already_int)
        @test Set(f_already_int_back.exp) == Set(f_already_int.exp)
        @test f_already_int_back.coeff == f_already_int.coeff
    end

    #==========================================================================
    # Type Consistency Tests
    ==========================================================================#
    @testset "Type Consistency" begin
        # Test 1: Rational{Int64}
        f_int64 = Signomial(
            [R(1), R(2)],
            [Rational{Int64}[1//2, 1//3], Rational{Int64}[1//4, 1//5]],
            false
        )
        g_int64 = TropicalNN.standardize(f_int64)
        @test g_int64.denominator isa Int64

        # Test 2: Rational{BigInt}
        f_bigint = Signomial(
            [R(1), R(2)],
            [Rational{BigInt}[1//2, 1//3], Rational{BigInt}[1//4, 1//5]],
            false
        )
        g_bigint = TropicalNN.standardize(f_bigint)
        @test g_bigint.denominator isa BigInt
    end

    #==========================================================================
    # Addition Tests
    ==========================================================================#
    @testset "Standardized Polynomial Addition" begin
        # Test 1: Basic addition with same denominator
        f = Signomial([R(1), R(2)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g = Signomial([R(3), R(4)], [[1//2, 0//1], [1//1, 0//1]]; sorted=false)
        f_std = TropicalNN.standardize(f)
        g_std = TropicalNN.standardize(g)
        h_std = f_std + g_std

        @test h_std isa TropicalNN.StandardizedTropicalPoly
        @test h_std.denominator == 2

        # Test 2: Addition with different denominators
        f2 = Signomial([R(1)], [[1//2, 0//1]]; sorted=false)
        g2 = Signomial([R(2)], [[0//1, 1//3]]; sorted=false)
        f2_std = TropicalNN.standardize(f2)
        g2_std = TropicalNN.standardize(g2)
        h2_std = f2_std + g2_std

        @test h2_std.denominator == 6  # lcm(2, 3)

        # Test 3: Verify correctness via destandardize
        h2_puiseux = TropicalNN.destandardize(h2_std)
        h2_direct = f2 + g2
        @test Set(h2_puiseux.exp) == Set(h2_direct.exp)
        @test h2_puiseux.coeff == h2_direct.coeff

        # Test 4: Commutativity - test via destandardize
        fwd_puiseux = TropicalNN.destandardize(f_std + g_std)
        rev_puiseux = TropicalNN.destandardize(g_std + f_std)
        @test Set(fwd_puiseux.exp) == Set(rev_puiseux.exp)
        @test fwd_puiseux.coeff == rev_puiseux.coeff
    end

    #==========================================================================
    # Multiplication Tests
    ==========================================================================#
    @testset "Standardized Polynomial Multiplication" begin
        # Test 1: Basic multiplication
        f = Signomial([R(1), R(2)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g = Signomial([R(3), R(4)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        f_std = TropicalNN.standardize(f)
        g_std = TropicalNN.standardize(g)
        h_std = f_std * g_std

        @test h_std isa TropicalNN.StandardizedTropicalPoly
        @test h_std.denominator == 2

        # Test 2: Verify correctness via destandardize
        h_puiseux = TropicalNN.destandardize(h_std)
        h_direct = f * g
        @test Set(h_puiseux.exp) == Set(h_direct.exp)
        for exp_vec in h_puiseux.exp
            @test h_puiseux.coeff[exp_vec] == h_direct.coeff[exp_vec]
        end

        # Test 3: Multiplication with different denominators
        f2 = Signomial([R(1), R(2)], [[1//2, 0//1], [0//1, 1//3]]; sorted=false)
        g2 = Signomial([R(3)], [[1//3, 0//1]]; sorted=false)
        f2_std = TropicalNN.standardize(f2)
        g2_std = TropicalNN.standardize(g2)
        h2_std = f2_std * g2_std

        @test h2_std.denominator == 6  # lcm(6, 3)

        # Test 4: Commutativity - test via destandardize
        h_fwd_puiseux = TropicalNN.destandardize(f_std * g_std)
        h_rev_puiseux = TropicalNN.destandardize(g_std * f_std)
        @test Set(h_fwd_puiseux.exp) == Set(h_rev_puiseux.exp)
        for exp_vec in h_fwd_puiseux.exp
            @test h_fwd_puiseux.coeff[exp_vec] == h_rev_puiseux.coeff[exp_vec]
        end
    end

    #==========================================================================
    # Quicksum Tests
    ==========================================================================#
    @testset "Standardized Quicksum" begin
        # Test 1: Basic quicksum with same denominator
        polys = [
            Signomial([R(1), R(2)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false),
            Signomial([R(3), R(4)], [[1//2, 0//1], [1//1, 0//1]]; sorted=false),
            Signomial([R(5)], [[0//1, 1//2]]; sorted=false)
        ]
        polys_std = [TropicalNN.standardize(p) for p in polys]
        result_std = TropicalNN.quicksum(polys_std)

        @test result_std isa TropicalNN.StandardizedTropicalPoly
        @test result_std.denominator == 2

        # Test 2: Verify correctness
        result_puiseux = TropicalNN.destandardize(result_std)
        result_direct = TropicalNN.quicksum(polys)
        @test Set(result_puiseux.exp) == Set(result_direct.exp)

        # Test 3: Quicksum with different denominators
        polys2 = [
            Signomial([R(1)], [[1//2, 0//1]]; sorted=false),
            Signomial([R(2)], [[1//3, 0//1]]; sorted=false),
            Signomial([R(3)], [[1//5, 0//1]]; sorted=false)
        ]
        polys2_std = [TropicalNN.standardize(p) for p in polys2]
        result2_std = TropicalNN.quicksum(polys2_std)

        @test result2_std.denominator == 30  # lcm(2, 3, 5)
    end

    #==========================================================================
    # Evaluation Tests
    ==========================================================================#
    @testset "Horner Evaluation" begin
        # Test 1: Univariate evaluation
        f_uni = Signomial([R(1), R(2), R(3)], [[2//1], [1//1], [0//1]]; sorted=false)
        f_uni_std = TropicalNN.standardize(f_uni)

        point = [R(2)]
        result_horner = TropicalNN.eval_horner(f_uni_std, point)
        result_direct = TropicalNN.evaluate(f_uni_std, point)
        result_puiseux = TropicalNN.evaluate(f_uni, point)

        @test result_horner == result_direct
        @test result_horner == result_puiseux

        # Test 2: Multivariate evaluation (falls back to direct)
        f_multi = Signomial(
            [R(1), R(2), R(3)],
            [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]],
            false
        )
        f_multi_std = TropicalNN.standardize(f_multi)

        point2 = [R(2), R(3)]
        result_multi_horner = TropicalNN.eval_horner(f_multi_std, point2)
        result_multi_direct = TropicalNN.evaluate(f_multi_std, point2)
        result_multi_puiseux = TropicalNN.evaluate(f_multi, point2)

        @test result_multi_horner == result_multi_direct
        @test result_multi_horner == result_multi_puiseux

        # Test 3: Direct evaluation with rational denominators
        f3 = Signomial([R(5), R(7)], [[1//2, 1//3], [1//4, 1//6]]; sorted=false)
        f3_std = TropicalNN.standardize(f3)

        point3 = [R(4), R(27)]
        result3_std = TropicalNN.evaluate(f3_std, point3)
        result3_puiseux = TropicalNN.evaluate(f3, point3)

        @test result3_std == result3_puiseux
    end

    #==========================================================================
    # Round-trip Arithmetic Consistency Tests
    ==========================================================================#
    @testset "Round-trip Arithmetic Consistency" begin
        # --- Addition round-trips ---

        # Addition 1: Two 2-term polys, same denominator (1//2)
        f_a1 = Signomial([R(1), R(3)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g_a1 = Signomial([R(2), R(4)], [[1//2, 1//2], [0//1, 1//1]]; sorted=false)
        rt_a1 = TropicalNN.destandardize(TropicalNN.standardize(f_a1) + TropicalNN.standardize(g_a1))
        direct_a1 = f_a1 + g_a1
        @test Set(rt_a1.exp) == Set(direct_a1.exp)
        @test rt_a1.coeff == direct_a1.coeff

        # Addition 2: Two 3-term polys, same denominator (1//4)
        f_a2 = Signomial(
            [R(1), R(2), R(5)],
            [[1//4, 0//1], [0//1, 3//4], [1//2, 1//4]],
            false
        )
        g_a2 = Signomial(
            [R(3), R(0), R(7)],
            [[3//4, 1//4], [1//4, 1//2], [0//1, 1//4]],
            false
        )
        rt_a2 = TropicalNN.destandardize(TropicalNN.standardize(f_a2) + TropicalNN.standardize(g_a2))
        direct_a2 = f_a2 + g_a2
        @test Set(rt_a2.exp) == Set(direct_a2.exp)
        @test rt_a2.coeff == direct_a2.coeff

        # Addition 3: Different denominators (1//2 vs 1//3)
        f_a3 = Signomial([R(2), R(6)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g_a3 = Signomial([R(1), R(4)], [[1//3, 0//1], [0//1, 2//3]]; sorted=false)
        rt_a3 = TropicalNN.destandardize(TropicalNN.standardize(f_a3) + TropicalNN.standardize(g_a3))
        direct_a3 = f_a3 + g_a3
        @test Set(rt_a3.exp) == Set(direct_a3.exp)
        @test rt_a3.coeff == direct_a3.coeff

        # Addition 4: Different denominators (1//3 vs 1//5), 3-term polys
        f_a4 = Signomial(
            [R(0), R(3), R(1)],
            [[1//3, 2//3], [2//3, 1//3], [0//1, 1//3]],
            false
        )
        g_a4 = Signomial(
            [R(2), R(5)],
            [[1//5, 0//1], [0//1, 3//5]],
            false
        )
        rt_a4 = TropicalNN.destandardize(TropicalNN.standardize(f_a4) + TropicalNN.standardize(g_a4))
        direct_a4 = f_a4 + g_a4
        @test Set(rt_a4.exp) == Set(direct_a4.exp)
        @test rt_a4.coeff == direct_a4.coeff

        # --- Multiplication round-trips ---

        # Multiplication 1: Two 2-term polys, same denominator
        f_m1 = Signomial([R(1), R(3)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g_m1 = Signomial([R(2), R(4)], [[1//2, 1//2], [0//1, 1//1]]; sorted=false)
        rt_m1 = TropicalNN.destandardize(TropicalNN.standardize(f_m1) * TropicalNN.standardize(g_m1))
        direct_m1 = f_m1 * g_m1
        @test Set(rt_m1.exp) == Set(direct_m1.exp)
        for exp_vec in rt_m1.exp
            @test rt_m1.coeff[exp_vec] == direct_m1.coeff[exp_vec]
        end

        # Multiplication 2: Different denominators (1//2 vs 1//3)
        f_m2 = Signomial([R(2), R(5)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g_m2 = Signomial([R(1), R(3)], [[1//3, 0//1], [0//1, 2//3]]; sorted=false)
        rt_m2 = TropicalNN.destandardize(TropicalNN.standardize(f_m2) * TropicalNN.standardize(g_m2))
        direct_m2 = f_m2 * g_m2
        @test Set(rt_m2.exp) == Set(direct_m2.exp)
        for exp_vec in rt_m2.exp
            @test rt_m2.coeff[exp_vec] == direct_m2.coeff[exp_vec]
        end

        # Multiplication 3: 3-term by 2-term, different denominators (1//4 vs 1//3)
        f_m3 = Signomial(
            [R(1), R(2), R(0)],
            [[1//4, 1//2], [1//2, 0//1], [0//1, 3//4]],
            false
        )
        g_m3 = Signomial([R(3), R(1)], [[1//3, 0//1], [0//1, 1//3]]; sorted=false)
        rt_m3 = TropicalNN.destandardize(TropicalNN.standardize(f_m3) * TropicalNN.standardize(g_m3))
        direct_m3 = f_m3 * g_m3
        @test Set(rt_m3.exp) == Set(direct_m3.exp)
        for exp_vec in rt_m3.exp
            @test rt_m3.coeff[exp_vec] == direct_m3.coeff[exp_vec]
        end

        # --- Quicksum round-trips ---

        # Quicksum 1: Three 2-term polys, same denominator
        f_q1 = Signomial([R(1), R(2)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g_q1 = Signomial([R(3), R(0)], [[1//1, 0//1], [0//1, 1//2]]; sorted=false)
        h_q1 = Signomial([R(4), R(1)], [[1//2, 1//2], [0//1, 1//1]]; sorted=false)
        rt_q1 = TropicalNN.destandardize(
            TropicalNN.quicksum([TropicalNN.standardize(f_q1), TropicalNN.standardize(g_q1), TropicalNN.standardize(h_q1)])
        )
        direct_q1 = TropicalNN.quicksum([f_q1, g_q1, h_q1])
        @test Set(rt_q1.exp) == Set(direct_q1.exp)
        @test rt_q1.coeff == direct_q1.coeff

        # Quicksum 2: Three polys with different denominators (1//2, 1//3, 1//5)
        f_q2 = Signomial([R(1), R(4)], [[1//2, 0//1], [0//1, 1//2]]; sorted=false)
        g_q2 = Signomial([R(2), R(3)], [[1//3, 0//1], [0//1, 2//3]]; sorted=false)
        h_q2 = Signomial([R(0), R(5)], [[1//5, 0//1], [0//1, 3//5]]; sorted=false)
        rt_q2 = TropicalNN.destandardize(
            TropicalNN.quicksum([TropicalNN.standardize(f_q2), TropicalNN.standardize(g_q2), TropicalNN.standardize(h_q2)])
        )
        direct_q2 = TropicalNN.quicksum([f_q2, g_q2, h_q2])
        @test Set(rt_q2.exp) == Set(direct_q2.exp)
        @test rt_q2.coeff == direct_q2.coeff

        # Quicksum 3: Three 3-term polys, mixed denominators (1//2, 1//3, 1//4)
        f_q3 = Signomial(
            [R(1), R(2), R(3)],
            [[1//2, 0//1], [0//1, 1//2], [1//2, 1//2]],
            false
        )
        g_q3 = Signomial(
            [R(4), R(0), R(2)],
            [[1//3, 0//1], [0//1, 1//3], [2//3, 1//3]],
            false
        )
        h_q3 = Signomial(
            [R(1), R(5), R(3)],
            [[1//4, 0//1], [0//1, 3//4], [1//2, 1//4]],
            false
        )
        rt_q3 = TropicalNN.destandardize(
            TropicalNN.quicksum([TropicalNN.standardize(f_q3), TropicalNN.standardize(g_q3), TropicalNN.standardize(h_q3)])
        )
        direct_q3 = TropicalNN.quicksum([f_q3, g_q3, h_q3])
        @test Set(rt_q3.exp) == Set(direct_q3.exp)
        @test rt_q3.coeff == direct_q3.coeff
    end
end
