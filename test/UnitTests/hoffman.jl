using Test, TropicalNN, Random, Oscar

@testset verbose = true "Hoffman" begin
    Random.seed!(42)

    @test round(exact_hoff([1 0 0; 0 1 0; 0 0 1]), digits = 2) == 1.0
    @test round(exact_hoff([1 0 0; 0 1 0; 0 0 1; -1 -1 -1]), digits = 2) == 3.0
    @test round(exact_hoff([1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]), digits = 2) ==
          1.0
    @test pvz_hoff([1 0 0; 0 1 0; 0 0 1])≈exact_hoff([1 0 0; 0 1 0; 0 0 1])
    @test pvz_hoff([1 0 0; 0 1 0; 0 0 1; -1 -1 -1]) ≈
          exact_hoff([1 0 0; 0 1 0; 0 0 1; -1 -1 -1])
    @test pvz_hoff([1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]) ≈
          exact_hoff([1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1])

    split_matrix = [1.0 0.0; -1.0 0.0; 0.0 1.0]
    h_pvz, F, I = pvz_hoff(split_matrix; return_certificates = true)
    @test h_pvz≈exact_hoff(split_matrix)
    @test !isempty(F)
    @test !isempty(I)

    R = tropical_semiring(max)
    single_monomial = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
    # Degenerate one-piece case: there is no finite A-surjectivity certificate
    # and no linear-region boundary, so these APIs use Inf as the convention.
    @test exact_hoff(single_monomial) == Inf
    @test pvz_hoff(single_monomial) == Inf
    @test upper_hoff(single_monomial) == Inf
    @test lower_hoff(single_monomial) == Inf
    @test exact_er(single_monomial) == Inf
    @test upper_er(single_monomial) == Inf

    tropical_zero = Signomial([zero(R(0))], [[0//1, 0//1]]; sorted = false)
    @test monomial_count(tropical_zero) == 0
    @test exact_hoff(tropical_zero) == Inf
    @test pvz_hoff(tropical_zero) == Inf
    @test upper_hoff(tropical_zero) == Inf
    @test lower_hoff(tropical_zero) == Inf
    @test exact_er(tropical_zero) == Inf
    @test upper_er(tropical_zero) == Inf

    empty_signomial = Signomial(
        TropicalNN._TROPICAL_COEFF[],
        Vector{Vector{Rational{BigInt}}}();
        sorted = false
    )
    @test exact_hoff(empty_signomial) == Inf
    @test pvz_hoff(empty_signomial) == Inf
    @test upper_hoff(empty_signomial) == Inf
    @test lower_hoff(empty_signomial) == Inf

    Random.seed!(42)
    mat = rand(3, 3)
    h_exact = exact_hoff(mat)
    h_pvz = pvz_hoff(mat)
    h_upper = upper_hoff(mat)
    h_lower = lower_hoff(mat)
    @test h_pvz≈h_exact
    @test h_exact <= h_upper
    @test h_exact >= h_lower

    Random.seed!(42)
    pmap = random_pmap(3, 3)
    h_exact = exact_hoff(pmap)
    h_pvz = pvz_hoff(pmap)
    h_upper = upper_hoff(pmap)
    h_lower = lower_hoff(pmap)
    @test h_pvz≈h_exact
    @test h_exact <= h_upper
    @test h_exact >= h_lower

    Random.seed!(42)
    w, b, t = random_mlp([2, 2, 1])
    rmap = mlp_to_trop(w, b, t)[1]
    h_exact = exact_hoff(rmap)
    h_pvz = pvz_hoff(rmap)
    h_upper = upper_hoff(rmap)
    h_lower = lower_hoff(rmap)
    @test h_pvz≈h_exact
    @test h_exact <= h_upper
    @test h_exact >= h_lower

    # effective radius tests

    Random.seed!(42)
    pmap = random_pmap(3, 3)
    er_exact = exact_er(pmap)
    er_upper = upper_er(pmap)
    @test er_exact <= er_upper

    Random.seed!(42)
    w, b, t = random_mlp([2, 2, 1])
    rmap = mlp_to_trop(w, b, t)[1]
    er_exact = exact_er(rmap)
    er_upper = upper_er(rmap)
    @test er_exact <= er_upper
end
