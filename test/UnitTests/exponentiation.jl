using Test, TropicalNN, Oscar

@testset verbose = true "Exponentiation methods" begin
    R = tropical_semiring(max)

    @testset verbose = true "Oscar TropicalSemiringElem ^ Rational" begin
        @test Float64(Rational(R(4) ^ (1//2))) == 2.0   # 4 * 1/2 = 2
        @test Float64(Rational(R(6) ^ (2//3))) == 4.0   # 6 * 2/3 = 4
        @test Float64(Rational(R(0) ^ (3//2))) == 0.0
        @test Float64(Rational(R(10) ^ (1//5))) == 2.0   # 10 * 1/5 = 2
        @test iszero(zero(R) ^ (3//2))
    end

    @testset verbose = true "Signomial ^ Int64" begin
        f = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)

        g2 = f ^ Int64(2)
        @test length(g2) == 2
        @test Float64(Rational(TropicalNN.get_coeff_by_exp(g2, Rational{Int64}[2, 0]))) ==
              2.0  # 1*2 = 2
        @test Float64(Rational(TropicalNN.get_coeff_by_exp(g2, Rational{Int64}[0, 2]))) ==
              4.0  # 2*2 = 4

        g0 = f ^ Int64(0)
        @test length(g0) == 1
        @test Float64(Rational(TropicalNN.get_coeff(g0, 1))) == 0.0

        empty = (zero(R(0)) * f) + (zero(R(0)) * f)
        @test length(empty ^ Int64(0)) == 1
        @test length(empty ^ Int64(2)) == 0
    end

    @testset verbose = true "Signomial ^ Float64" begin
        float_exponent = Signomial([R(0)], [[0.5]]; sorted = true)
        @test TropicalNN.evaluate(float_exponent, [R(4)]) == R(2)
    end

    @testset verbose = true "Signomial ^ Rational" begin
        f = Signomial([R(2), R(4)], [[2//1, 0//1], [0//1, 2//1]]; sorted = false)

        g = f ^ (1//2)
        @test length(g) == 2
        @test any(e -> e == Rational{BigInt}[1, 0], TropicalNN.exponents(g))
        @test any(e -> e == Rational{BigInt}[0, 1], TropicalNN.exponents(g))
        exp1 = Rational{BigInt}[1, 0]
        @test Float64(Rational(TropicalNN.get_coeff_by_exp(g, exp1))) == 1.0
        exp2 = Rational{BigInt}[0, 1]
        @test Float64(Rational(TropicalNN.get_coeff_by_exp(g, exp2))) == 2.0

        g0 = f ^ (0//1)
        @test length(g0) == 1

        f_max = Signomial([R(0), R(0)], [[0//1], [1//1]]; sorted = false)
        @test_throws DomainError f_max ^ (-1//1)
        negative_integer = -1
        @test_throws DomainError f_max ^ negative_integer
        @test_throws MethodError f_max ^ (-1)
        @test_throws DomainError f_max ^ (-1.0)
        @test_throws MethodError inv(f_max)

        integer_exponents = Signomial([R(0)], [[1]]; sorted = false)
        half_power = integer_exponents ^ (1//2)
        @test TropicalNN.exponents(half_power) == [[1//2]]
    end

    @testset verbose = true "RationalSignomial ^ Int64" begin
        num = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        den = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        q = RationalSignomial(num, den)

        qinv = q ^ Int64(-1)
        @test qinv.num == den
        @test qinv.den == num
    end

    @testset verbose = true "RationalSignomial ^ Float64" begin
        num = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        den = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        q = RationalSignomial(num, den)

        qinv = q ^ (-1.0)
        @test qinv.num == den
        @test qinv.den == num
    end

    @testset verbose = true "RationalSignomial ^ Rational" begin
        num = Signomial([R(2), R(4)], [[2//1, 0//1], [0//1, 2//1]]; sorted = false)
        den = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        q = RationalSignomial(num, den)

        qinv = q ^ (-1//1)
        q_value = TropicalNN.evaluate(q, [R(2), R(1)])
        @test TropicalNN.evaluate(qinv, [R(2), R(1)]) == one(q_value) / q_value
        @test TropicalNN.evaluate(inv(q), [R(2), R(1)]) == one(q_value) / q_value
    end
end
