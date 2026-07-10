using Test, TropicalNN, Random, Oscar

@testset verbose = true "Hoffman" begin
    function certificates_cover_nonempty_subsets(m, F, I)
        for mask in 1:(2 ^ m - 1)
            subset = [index for index in 1:m if !iszero(mask & (1 << (index - 1)))]
            covered_by_surjective = any(F_set -> issubset(subset, F_set), F)
            covered_by_nonsurjective = any(I_set -> issubset(I_set, subset), I)
            (covered_by_surjective || covered_by_nonsurjective) || return false
        end
        return true
    end

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
    @test certificates_cover_nonempty_subsets(size(split_matrix, 1), F, I)

    zero_matrix = [0.0;;]
    h_pvz, F, I = pvz_hoff(zero_matrix; return_certificates = true)
    @test exact_hoff(zero_matrix) == 0.0
    @test h_pvz == 0.0
    @test upper_hoff(zero_matrix) == 0.0
    @test lower_hoff(zero_matrix) == 0.0
    @test certificates_cover_nonempty_subsets(size(zero_matrix, 1), F, I)

    small_matrix = [1e-12;;]
    @test surjectivity_test(small_matrix)[2] > 0
    @test exact_hoff(small_matrix)≈1e12
    @test pvz_hoff(small_matrix)≈1e12

    non_full_row_rank_surjective = ones(2, 1)
    h_pvz, F, I = pvz_hoff(non_full_row_rank_surjective; return_certificates = true)
    @test h_pvz≈exact_hoff(non_full_row_rank_surjective)
    @test [1, 2] in F
    @test isempty(I)

    R = tropical_semiring(max)
    single_monomial = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
    # Degenerate one-piece case: the paper convention initializes H at 0,
    # and the minimal effective radius of a single affine region is 0.
    @test exact_hoff(single_monomial) == 0.0
    @test pvz_hoff(single_monomial) == 0.0
    @test upper_hoff(single_monomial) == 0.0
    @test lower_hoff(single_monomial) == 0.0
    @test exact_er(single_monomial) == 0.0
    @test upper_er(single_monomial) == 0.0

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
