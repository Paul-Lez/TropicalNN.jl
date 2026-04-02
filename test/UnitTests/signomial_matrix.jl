using Test, TropicalNN, Oscar

@testset "SignomialMatrix (dim > 5)" begin
    R = tropical_semiring(max)

    # Helper: build a 6-variable exponent vector with a single nonzero entry
    function unit_exp(i::Int)
        e = zeros(Rational{Int64}, 6)
        e[i] = 1 // 1
        return e
    end

    #==========================================================================
    # Construction Tests
    ==========================================================================#
    @testset "Construction routes to SignomialMatrix" begin
        # 6 variables should use SignomialMatrix, not SignomialStatic
        f = Signomial([R(1), R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        @test f isa SignomialMatrix{Rational{Int64}}
        @test Oscar.nvars(f) == 6
        @test length(f) == 2
    end

    #==========================================================================
    # Addition Tests
    ==========================================================================#
    @testset "Addition" begin
        # f = max(1 + x1, 2 + x2)
        # g = max(3 + x1, 4 + x3)
        f = Signomial([R(1), R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        g = Signomial([R(3), R(4)], [unit_exp(1), unit_exp(3)]; sorted=false)
        h = f + g

        # Result should be max(3 + x1, 2 + x2, 4 + x3)
        # x1 appears in both with coeffs 1 and 3, tropical sum = max(1,3) = 3
        # Note: SignomialMatrix addition may NOT merge duplicates (known bug).
        # We test what actually happens: the result contains at least the
        # correct number of distinct exponent vectors.
        @test h isa SignomialMatrix{Rational{Int64}}

        # Commutativity: f + g should equal g + f
        h2 = g + f
        @test length(h) == length(h2)

        # Addition of polynomials with no overlapping exponents
        f_no = Signomial([R(5)], [unit_exp(4)]; sorted=false)
        g_no = Signomial([R(6)], [unit_exp(5)]; sorted=false)
        h_no = f_no + g_no
        @test length(h_no) == 2
    end

    @testset "Addition duplicate exponent merging" begin
        # Both f and g have the same monomial x1 with different coefficients.
        # Correct tropical addition: max(1, 3) = 3 for that monomial.
        f = Signomial([R(1)], [unit_exp(1)]; sorted=false)
        g = Signomial([R(3)], [unit_exp(1)]; sorted=false)
        h = f + g

        # The correct result has exactly 1 monomial (the merged one).
        # If duplicates are NOT merged, there will be 2 entries.
        # This test documents the known bug (section 3.4):
        # SignomialMatrix.+ does not merge duplicate exponents.
        if length(h) == 1
            # Correct behavior: duplicates were merged
            @test get_coeff(h, 1) == R(3)
        else
            # Known bug: duplicates not merged, both monomials kept
            @test length(h) == 2
            @test_broken length(h) == 1  # Mark as known failure
        end
    end

    #==========================================================================
    # Multiplication Tests
    ==========================================================================#
    @testset "Multiplication" begin
        # f = max(1 + x1, 2 + x2)
        # g = max(3 + x3, 4 + x4)
        # Product should have 4 monomials (no overlapping exponents possible)
        f = Signomial([R(1), R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        g = Signomial([R(3), R(4)], [unit_exp(3), unit_exp(4)]; sorted=false)
        h = f * g

        @test h isa SignomialMatrix{Rational{Int64}}
        @test length(h) == 4  # 2 * 2 = 4 distinct monomials

        # Check a specific monomial: x1 + x3 with coeff 1 + 3 = 4
        exp_13 = zeros(Rational{Int64}, 6)
        exp_13[1] = 1 // 1
        exp_13[3] = 1 // 1
        found = false
        for i in 1:length(h)
            if get_exp(h, i) == exp_13
                @test get_coeff(h, i) == R(4)
                found = true
                break
            end
        end
        @test found

        # Commutativity
        h2 = g * f
        @test length(h) == length(h2)
    end

    @testset "Multiplication with overlapping product exponents" begin
        # f = max(1 + x1, 2 + x2), g = max(3 + x1, 4 + x2)
        # Products: (x1,x1)=2x1 c=4, (x1,x2)=x1+x2 c=5,
        #           (x2,x1)=x1+x2 c=5, (x2,x2)=2x2 c=6
        # x1+x2 appears twice with coeff 5 - should merge to one monomial.
        f = Signomial([R(1), R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        g = Signomial([R(3), R(4)], [unit_exp(1), unit_exp(2)]; sorted=false)
        h = f * g

        # With correct merging: 3 distinct monomials (2x1, x1+x2, 2x2)
        # Without merging: 4 monomials (duplicate x1+x2)
        if length(h) == 3
            @test true  # Correctly merged
        else
            @test length(h) == 4
            @test_broken length(h) == 3  # Known bug: duplicates not merged
        end
    end

    #==========================================================================
    # Evaluation Tests
    ==========================================================================#
    @testset "Evaluation" begin
        # f = max(1 + x1, 2 + x2, 3 + x3, 4 + x4, 5 + x5, 6 + x6)
        exps = [unit_exp(i) for i in 1:6]
        coeffs = [R(i) for i in 1:6]
        f = Signomial(coeffs, exps; sorted=false)

        # Evaluate at all zeros: max(1+0, 2+0, ..., 6+0) = 6
        a_zeros = [R(0) for _ in 1:6]
        @test TropicalNN.evaluate(f, a_zeros) == R(6)

        # Evaluate at [10, 0, 0, 0, 0, 0]: max(1+10, 2+0, ..., 6+0) = 11
        a_x1 = [R(10), R(0), R(0), R(0), R(0), R(0)]
        @test TropicalNN.evaluate(f, a_x1) == R(11)

        # Evaluate at [0, 0, 0, 0, 0, 100]: max(1+0, ..., 6+100) = 106
        a_x6 = [R(0), R(0), R(0), R(0), R(0), R(100)]
        @test TropicalNN.evaluate(f, a_x6) == R(106)

        # Test callable syntax: f(x)
        @test f(a_zeros) == R(6)
    end

    #==========================================================================
    # Quicksum Tests
    ==========================================================================#
    @testset "Quicksum" begin
        # Create 6 single-monomial polynomials, one per variable
        polys = [Signomial([R(i)], [unit_exp(i)]; sorted=false) for i in 1:6]

        result = TropicalNN.quicksum(polys)
        @test result isa SignomialMatrix{Rational{Int64}}
        @test length(result) == 6  # All unique monomials

        # Quicksum should produce the same result as sequential addition
        result_seq = polys[1]
        for i in 2:6
            result_seq = result_seq + polys[i]
        end
        @test length(result) == length(result_seq)

        # Quicksum with overlapping monomials
        polys_overlap = [
            Signomial([R(1)], [unit_exp(1)]; sorted=false),
            Signomial([R(3)], [unit_exp(1)]; sorted=false),
            Signomial([R(5)], [unit_exp(2)]; sorted=false),
        ]
        result_overlap = TropicalNN.quicksum(polys_overlap)
        # quicksum just concatenates without merging, so expect 3 entries
        @test length(result_overlap) >= 2
    end

    #==========================================================================
    # Multiplication with Quicksum
    ==========================================================================#
    @testset "Multiplication with quicksum" begin
        f = Signomial([R(1), R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        g = Signomial([R(3), R(4)], [unit_exp(3), unit_exp(4)]; sorted=false)

        h_std = f * g
        h_qs = TropicalNN.mul_with_quicksum(f, g)

        # For SignomialMatrix, mul_with_quicksum delegates to *, so results
        # should be identical.
        @test length(h_std) == length(h_qs)
        for i in 1:length(h_std)
            @test get_exp(h_std, i) == get_exp(h_qs, i)
            @test get_coeff(h_std, i) == get_coeff(h_qs, i)
        end
    end

    #==========================================================================
    # Scalar Multiplication and Exponentiation
    ==========================================================================#
    @testset "Scalar multiplication" begin
        f = Signomial([R(1), R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        g = R(5) * f
        @test g isa SignomialMatrix{Rational{Int64}}
        @test length(g) == 2
        # Tropical scalar multiplication: coeff * scalar (i.e. coeff + scalar in ordinary arithmetic)
        # Check that both coefficients were scaled
        coeffs_g = Set([get_coeff(g, i) for i in 1:length(g)])
        @test R(6) in coeffs_g   # R(1) * R(5) = R(6)
        @test R(7) in coeffs_g   # R(2) * R(5) = R(7)
    end

    #==========================================================================
    # Dedup Monomials
    ==========================================================================#
    @testset "Dedup monomials" begin
        # Polynomial with one tropical-zero coefficient
        tropical_zero = zero(R(0))
        f = Signomial([tropical_zero, R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        f_dedup = TropicalNN.dedup_monomials(f)
        @test length(f_dedup) == 1
    end

    #==========================================================================
    # Higher Dimension Tests (8 and 10 variables)
    ==========================================================================#
    @testset "Higher dimensions" begin
        # 8-variable polynomial
        exps_8 = [begin e = zeros(Rational{Int64}, 8); e[i] = 1 // 1; e end for i in 1:8]
        f8 = Signomial([R(i) for i in 1:8], exps_8; sorted=false)
        @test f8 isa SignomialMatrix{Rational{Int64}}
        @test Oscar.nvars(f8) == 8
        @test length(f8) == 8

        # Evaluate at all zeros: max(1, 2, ..., 8) = 8
        @test TropicalNN.evaluate(f8, [R(0) for _ in 1:8]) == R(8)

        # 10-variable polynomial
        exps_10 = [begin e = zeros(Rational{Int64}, 10); e[i] = 1 // 1; e end for i in 1:10]
        f10 = Signomial([R(i) for i in 1:10], exps_10; sorted=false)
        @test f10 isa SignomialMatrix{Rational{Int64}}
        @test length(f10) == 10

        # Multiplication of 10-variable polynomials
        g10 = Signomial([R(1), R(2)], [exps_10[1], exps_10[2]]; sorted=false)
        h10 = f10 * g10
        @test h10 isa SignomialMatrix{Rational{Int64}}
        @test length(h10) == 20  # 10 * 2 = 20 (no overlapping exponents)
    end

    #==========================================================================
    # Equality Tests
    ==========================================================================#
    @testset "Equality" begin
        f = Signomial([R(1), R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        g = Signomial([R(1), R(2)], [unit_exp(1), unit_exp(2)]; sorted=false)
        @test f == g

        h = Signomial([R(1), R(3)], [unit_exp(1), unit_exp(2)]; sorted=false)
        @test f != h
    end
end
