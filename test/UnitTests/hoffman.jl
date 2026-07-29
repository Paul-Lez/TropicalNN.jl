using Test, TropicalNN, Random, Oscar

@testset verbose = true "Hoffman" begin
    Random.seed!(42)

    @test round(hoffman_constant([1 0 0; 0 1 0; 0 0 1]), digits = 2) == 1.0
    @test round(hoffman_constant([1 0 0; 0 1 0; 0 0 1; -1 -1 -1]), digits = 2) == 3.0
    @test round(hoffman_constant([1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]), digits = 2) ==
          1.0
    @test hoffman_constant([1 0 0; 0 1 0; 0 0 1]) ≈
          hoffman_constant([1 0 0; 0 1 0; 0 0 1]; brute_force = true)
    @test hoffman_constant([1 0 0; 0 1 0; 0 0 1; -1 -1 -1]) ≈
          hoffman_constant([1 0 0; 0 1 0; 0 0 1; -1 -1 -1]; brute_force = true)
    @test hoffman_constant([1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1]) ≈
          hoffman_constant(
        [1 0 0; 0 1 0; 0 0 1; -1 0 0; 0 -1 0; 0 0 -1];
        brute_force = true
    )

    split_matrix = [1.0 0.0; -1.0 0.0; 0.0 1.0]
    h_pvz, F, I = TropicalNN._pvz_hoff(split_matrix; return_certificates = true)
    @test h_pvz ≈ hoffman_constant(split_matrix; brute_force = true)
    @test !isempty(F)
    @test !isempty(I)

    R = tropical_semiring(max)
    single_monomial = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
    # Degenerate one-piece case: there is no finite A-surjectivity certificate
    # and no linear-region boundary, so these APIs use Inf as the convention.
    @test hoffman_constant(single_monomial) == Inf
    @test hoffman_constant(single_monomial; brute_force = true) == Inf
    @test upper_hoffman_constant(single_monomial) == Inf
    @test lower_hoffman_constant(single_monomial) == Inf
    @test exact_er(single_monomial) == Inf
    @test upper_er(single_monomial) == Inf

    tropical_zero = Signomial([zero(R(0))], [[0//1, 0//1]]; sorted = false)
    @test monomial_count(tropical_zero) == 0
    @test hoffman_constant(tropical_zero) == Inf
    @test hoffman_constant(tropical_zero; brute_force = true) == Inf
    @test upper_hoffman_constant(tropical_zero) == Inf
    @test lower_hoffman_constant(tropical_zero) == Inf
    @test exact_er(tropical_zero) == Inf
    @test upper_er(tropical_zero) == Inf

    empty_signomial = Signomial(
        TropicalNN._TROPICAL_COEFF[],
        Vector{Vector{Rational{BigInt}}}();
        sorted = false
    )
    @test hoffman_constant(empty_signomial) == Inf
    @test hoffman_constant(empty_signomial; brute_force = true) == Inf
    @test upper_hoffman_constant(empty_signomial) == Inf
    @test lower_hoffman_constant(empty_signomial) == Inf

    Random.seed!(42)
    mat = rand(3, 3)
    h_exact = hoffman_constant(mat; brute_force = true)
    h_pvz = hoffman_constant(mat)
    h_upper = upper_hoffman_constant(mat)
    h_lower = lower_hoffman_constant(mat)
    @test h_pvz≈h_exact
    @test h_exact <= h_upper
    @test h_exact >= h_lower

    Random.seed!(42)
    pmap = random_signomial(3, 3)
    h_exact = hoffman_constant(pmap; brute_force = true)
    h_pvz = hoffman_constant(pmap)
    h_upper = upper_hoffman_constant(pmap)
    h_lower = lower_hoffman_constant(pmap)
    @test h_pvz≈h_exact
    @test h_exact <= h_upper
    @test h_exact >= h_lower

    Random.seed!(42)
    w, b, t = random_mlp([2, 2, 1])
    rmap = mlp_to_trop(w, b, t)[1]
    h_exact = hoffman_constant(rmap; brute_force = true)
    h_pvz = hoffman_constant(rmap)
    h_upper = upper_hoffman_constant(rmap)
    h_lower = lower_hoffman_constant(rmap)
    @test h_pvz≈h_exact
    @test h_exact <= h_upper
    @test h_exact >= h_lower

    # effective radius tests

    Random.seed!(42)
    pmap = random_signomial(3, 3)
    er_exact = exact_er(pmap)
    er_upper = upper_er(pmap)
    @test isapprox(er_exact, exact_er(pmap; brute_force = true); rtol = 1e-10)
    @test er_exact <= er_upper

    Random.seed!(42)
    w, b, t = random_mlp([2, 2, 1])
    rmap = mlp_to_trop(w, b, t)[1]
    er_exact = exact_er(rmap)
    er_upper = upper_er(rmap)
    @test isapprox(er_exact, exact_er(rmap; brute_force = true); rtol = 1e-10)
    @test er_exact <= er_upper
end
