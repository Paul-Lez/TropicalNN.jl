using Test, TropicalNN, Oscar

@testset verbose = true "_linearmap_matrices" begin
    R = tropical_semiring(max)

    @testset verbose = true "Signomial" begin
        f = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        A, b = TropicalNN._linearmap_matrices(f)

        @test size(A, 2) == 2
        @test size(A, 1) == length(b)
        @test size(A, 1) == 2

        rows = Set([Tuple(A[i, :]) for i in 1:size(A, 1)])
        @test (1.0, 0.0) in rows
        @test (0.0, 1.0) in rows

        @test length(b) == 2
    end

    @testset verbose = true "Single-monomial Signomial" begin
        f_const = Signomial([R(5)], [[0//1, 0//1]]; sorted = false)
        A, b = TropicalNN._linearmap_matrices(f_const)
        @test size(A, 1) == 1
        @test size(A, 2) == 2
        @test length(b) == 1
    end

    @testset verbose = true "Redundant monomials are retained" begin
        f = Signomial(
            [R(0), R(-1), R(0)],
            [[0//1], [1//1], [2//1]];
            sorted = false
        )
        A, b = TropicalNN._linearmap_matrices(f)

        @test vec(A) == [0.0, 1.0, 2.0]
        @test length(b) == 3

        f_pruned = TropicalNN.prune(f; parallel = false)
        A_pruned, b_pruned = TropicalNN._linearmap_matrices(f_pruned)
        @test vec(A_pruned) == [0.0, 2.0]
        @test length(b_pruned) == 2

        den = Signomial([R(0)], [[0//1]]; sorted = false)
        q = RationalSignomial(f, den)
        (Anum, Aden), (bnum, bden) = TropicalNN._linearmap_matrices(q)
        @test vec(Anum) == [0.0, 1.0, 2.0]
        @test vec(Aden) == [0.0]
        @test length(bnum) == 3
        @test length(bden) == 1
    end

    @testset verbose = true "RationalSignomial" begin
        num = Signomial([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]]; sorted = false)
        den = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        q = RationalSignomial(num, den)

        (Anum, Aden), (bnum, bden) = TropicalNN._linearmap_matrices(q)

        @test size(Anum, 2) == 2
        @test size(Anum, 1) == 2
        @test size(Aden, 2) == 2
        @test size(Aden, 1) == 1
        @test length(bnum) == 2
        @test length(bden) == 1
    end
end
