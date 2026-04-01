using Test, TropicalNN, Oscar

@testset "Exponentiation methods" begin
    R = tropical_semiring(max)

    @testset "TropicalSemiringElem ^ TropicalSemiringElem" begin
        a = R(3)
        b = R(2)
        @test Float64(Rational(a ^ b)) == 6.0   # 3*2 = 6
        @test Float64(Rational(R(0) ^ R(4))) == 0.0
        @test Float64(Rational(R(5) ^ R(1))) == 5.0
    end

    @testset "TropicalSemiringElem ^ Rational" begin
        @test Float64(Rational(R(4)  ^ (1//2))) == 2.0   # 4 * 1/2 = 2
        @test Float64(Rational(R(6)  ^ (2//3))) == 4.0   # 6 * 2/3 = 4
        @test Float64(Rational(R(0)  ^ (3//2))) == 0.0
        @test Float64(Rational(R(10) ^ (1//5))) == 2.0   # 10 * 1/5 = 2
    end

    @testset "TropicalSemiringElem ^ Float64" begin
        @test Float64(Rational(R(4) ^ 2.0)) ≈ 8.0    # 4*2 = 8
        @test Float64(Rational(R(3) ^ 0.5)) ≈ 1.5    # 3*0.5 = 1.5
        @test Float64(Rational(R(0) ^ 3.0)) ≈ 0.0
    end

    @testset "Signomial ^ Int64" begin
        f = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)

        g2 = f ^ Int64(2)
        @test g2 isa Signomial
        @test length(g2.exp) == 2
        @test haskey(g2.coeff, Rational{Int64}[2, 0])
        @test haskey(g2.coeff, Rational{Int64}[0, 2])
        @test Float64(Rational(g2.coeff[Rational{Int64}[2, 0]])) == 2.0  # 1*2 = 2
        @test Float64(Rational(g2.coeff[Rational{Int64}[0, 2]])) == 4.0  # 2*2 = 4

        g0 = f ^ Int64(0)
        @test length(g0.exp) == 1
        @test Float64(Rational(g0.coeff[first(g0.exp)])) == 0.0
    end

    @testset "Signomial ^ Float64" begin
        f = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)

        g = f ^ 2.0
        @test g isa Signomial
        @test length(g.exp) == 2

        g0 = f ^ 0.0
        @test length(g0.exp) == 1
    end

    @testset "Signomial ^ Rational" begin
        f = Signomial([R(2), R(4)], [[2//1, 0//1], [0//1, 2//1]]; sorted=false)

        g = f ^ (1//2)
        @test g isa Signomial
        @test length(g.exp) == 2
        @test any(e -> e == Rational{BigInt}[1, 0], g.exp)
        @test any(e -> e == Rational{BigInt}[0, 1], g.exp)
        exp1 = Rational{BigInt}[1, 0]
        @test Float64(Rational(g.coeff[exp1])) == 1.0
        exp2 = Rational{BigInt}[0, 1]
        @test Float64(Rational(g.coeff[exp2])) == 2.0

        g0 = f ^ (0//1)
        @test length(g0.exp) == 1
    end

    @testset "RationalSignomial ^ Int64" begin
        num = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        den = Signomial([R(0)],        [[0//1, 0//1]]; sorted=false)
        q = RationalSignomial(num, den)

        q2 = q ^ Int64(2)
        @test q2 isa RationalSignomial
        @test length(q2.num.exp) == 2

        q0 = q ^ Int64(0)
        @test q0 isa RationalSignomial
        @test length(q0.num.exp) == 1
    end

    @testset "RationalSignomial ^ Float64" begin
        num = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        den = Signomial([R(0)],        [[0//1, 0//1]]; sorted=false)
        q = RationalSignomial(num, den)

        qf = q ^ 2.0
        @test qf isa RationalSignomial
        @test length(qf.num.exp) == 2

        q0 = q ^ 0.0
        @test q0 isa RationalSignomial
        @test length(q0.num.exp) == 1
    end

    @testset "RationalSignomial ^ Rational" begin
        num = Signomial([R(2), R(4)], [[2//1, 0//1], [0//1, 2//1]]; sorted=false)
        den = Signomial([R(0)],        [[0//1, 0//1]]; sorted=false)
        q = RationalSignomial(num, den)

        qr = q ^ (1//2)
        @test qr isa RationalSignomial
        @test length(qr.num.exp) == 2

        q0 = q ^ (0//1)
        @test q0 isa RationalSignomial
        @test length(q0.num.exp) == 1
    end
end
