using Test, TropicalNN, Oscar

@testset "linearmap_matrices" begin
    R = tropical_semiring(max)

    @testset "Signomial" begin
        f = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        A, b = linearmap_matrices(f)

        @test size(A, 2) == 2
        @test size(A, 1) == length(b)
        @test size(A, 1) == 2

        rows = Set([Tuple(A[i, :]) for i in 1:size(A,1)])
        @test (1.0, 0.0) in rows
        @test (0.0, 1.0) in rows

        @test length(b) == 2
    end

    @testset "Single-monomial Signomial" begin
        f_const = Signomial([R(5)], [[0//1, 0//1]]; sorted=false)
        A, b = linearmap_matrices(f_const)
        @test size(A, 1) == 1
        @test size(A, 2) == 2
        @test length(b) == 1
    end

    @testset "RationalSignomial" begin
        num = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
        den = Signomial([R(0)],        [[0//1, 0//1]]; sorted=false)
        q = RationalSignomial(num, den)

        (Anum, Aden), (bnum, bden) = linearmap_matrices(q)

        @test size(Anum, 2) == 2
        @test size(Anum, 1) == 2
        @test size(Aden, 2) == 2
        @test size(Aden, 1) == 1
        @test length(bnum) == 2
        @test length(bden) == 1
    end
end

@testset "enum_linear_regions_rat — constant function" begin
    R = tropical_semiring(max)

    f_const = Signomial([R(3)], [[0//1, 0//1]]; sorted=false)
    g_const = Signomial([R(0)], [[0//1, 0//1]]; sorted=false)
    lr = enum_linear_regions_rat(f_const / g_const)

    @test length(lr) == 1
    @test length(lr[1].regions) == 1
    @test Oscar.is_feasible(lr[1].regions[1])

    f2 = Signomial([R(7)], [[0//1, 0//1]]; sorted=false)
    g2 = Signomial([R(2)], [[0//1, 0//1]]; sorted=false)
    lr2 = enum_linear_regions_rat(f2 / g2)
    @test length(lr2) == 1
end
