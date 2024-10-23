using Test, TropicalNN

@testset "Hoffman" begin
    @test round(exact_hoff([1 0 0;0 1 0;0 0 1]),digits=2)==1.0
    @test round(exact_hoff([1 0 0;0 1 0;0 0 1;-1 -1 -1]),digits=2)==3.0
    @test round(exact_hoff([1 0 0;0 1 0;0 0 1;-1 0 0;0 -1 0;0 0 -1]),digits=2)==1.0

    mat=rand(3,3)
    h_exact=exact_hoff(mat)
    h_upper=upper_hoff(mat)
    h_lower=lower_hoff(mat)
    @test h_exact<=h_upper
    @test h_exact>=h_lower

    pmap=random_pmap(3,3)
    h_exact=exact_hoff(pmap)
    h_upper=upper_hoff(pmap)
    h_lower=lower_hoff(pmap)
    @test h_exact<=h_upper
    @test h_exact>=h_lower

    w,b,t=random_mlp([2,2,1])
    rmap=mlp_to_trop_with_quicksum_with_strong_elim(w,b,t)[1]
    h_exact=exact_hoff(rmap)
    h_upper=upper_hoff(rmap)
    h_lower=lower_hoff(rmap)
    @test h_exact<=h_upper
    @test h_exact>=h_lower

    # effective radius tests

    pmap=random_pmap(3,3)
    er_exact=exact_er(pmap)
    er_upper=upper_er(pmap)
    @test er_exact<=er_upper

    w,b,t=random_mlp([2,2,1])
    rmap=mlp_to_trop_with_quicksum_with_strong_elim(w,b,t)[1]
    er_exact=exact_er(rmap)
    er_upper=upper_er(rmap)
    @test er_exact<=er_upper
    
end