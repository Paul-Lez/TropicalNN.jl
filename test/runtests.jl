using TropicalNN
using Test
using Oscar

@testset "TropicalNN.jl" begin
    w, b, t = TropicalNN.random_mlp([3, 2, 1])
    trop = mlp_to_trop(w, b, t)[1]
    @show length(enum_linear_regions_rat(trop))
    R = tropical_semiring(max)
    f = TropicalPuiseuxPoly([R(1), R(2), R(3)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
    # Write down the tropical polynomial 0*X^1*Y^7 + 4*X^0*Y^1 + (-5)*X^9*Y^1
    g = TropicalPuiseuxPoly([R(0), R(4), R(-5)], [[1//1, 7//1], [0//1, 1//1], [9//1, 1//1]], false) 

    # test addition and multiplication
    @test f + g == TropicalPuiseuxPoly{Rational{Int64}}(Dict{Any, Any}(Rational{Int64}[1, 1] => (3), Rational{Int64}[1, 0] => (1), Rational{Int64}[1, 7] => (0), Rational{Int64}[9, 1] => (-5), Rational{Int64}[0, 1] => (4)), Vector{Rational{Int64}}[[0, 1], [1, 0], [1, 1], [1, 7], [9, 1]]) 
    @test f * g == TropicalPuiseuxPoly{Rational{Int64}}(Dict{Any, Any}(Rational{Int64}[2, 8] => (3), Rational{Int64}[9, 2] => (-3), Rational{Int64}[1, 8] => (2), Rational{Int64}[10, 1] => (-4), Rational{Int64}[0, 2] => (6), Rational{Int64}[1, 1] => (5), Rational{Int64}[10, 2] => (-2), Rational{Int64}[2, 7] => (1), Rational{Int64}[1, 2] => (7)), Vector{Rational{Int64}}[[0, 2], [1, 1], [1, 2], [1, 8], [2, 7], [2, 8], [9, 2], [10, 1], [10, 2]])


    # test components
    V = [1, 2, 3, 4]
    D = Dict{Tuple{Int, Int}, Bool}((1, 2) => true, (3, 4) => true, (2, 3) => false)
    @test TropicalNN.components(V, D) == [[1, 2], [3, 4]]

    # test linear regions enumeration
    # Take u to be the tropical polynomial 0*X^1*Y^1 + 0*X^0*Y^1 = max(X, Y)
    u = TropicalPuiseuxPoly([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]], false)
    # and v to be the tropical polynomial 0 
    v = TropicalPuiseuxPoly([R(0)], [[0//1, 0//1]], false)
    @test length(enum_linear_regions_rat(u / v)) == 2

    # Monomial elimination
    # Take u to be the tropical polynomial max(0, x, 2x)
    u = TropicalPuiseuxPoly([R(0), R(0), R(0)], [[0//1], [1//1], [2//1]], false)
    # The monomial elimination of u should be max(0, 2x) since x is redundant
    @test monomial_strong_elim(u) == TropicalPuiseuxPoly([R(0), R(0)], [[0//1], [2//1]], false)

    # TODO: add tests for mlp_to_trop functions and the rest of the tropical algebra functions

    # visualisation tests

    # random_pmap
    pmap=random_pmap(2,4)
    @test typeof(pmap)<: TropicalPuiseuxPoly{Rational{BigInt}}
    @test nvars(pmap)==2
    @test monomial_count(pmap)==4

    # one-dimensional pmap 0*T^1 + 0*T^2 + -1*T^3
    pmap=TropicalPuiseuxPoly(Rational{BigInt}.([0,0,-1]),[Rational{BigInt}.([1]),Rational{BigInt}.([2]),Rational{BigInt}.([3])],false)

    reps=m_reps(pmap)
    @test reps==Dict{String, Vector{Any}}("m_reps" => [Array{Float64}[[0.0; 1.0; 2.0;;], [0.0, 0.0, 1.0]], Array{Float64}[[-1.0; 0.0; 1.0;;], [0.0, 0.0, 1.0]], Array{Float64}[[-2.0; -1.0; 0.0;;], [-1.0, -1.0, 0.0]]], "f_indices" => [1, 2, 3])

    bounding_box=get_full_bounding_box(pmap,reps)
    @test bounding_box==Dict(1 => [-1.0, 2.0])

    reps=bound_reps(reps,bounding_box)
    @test reps==Dict{String, Vector}("m_reps" => Vector{Array{Float64}}[[[0.0; 1.0; 2.0; 1.0; -1.0;;], [0.0, 0.0, 1.0, 2.0, 1.0]], [[-1.0; 0.0; 1.0; 1.0; -1.0;;], [0.0, 0.0, 1.0, 2.0, 1.0]], [[-2.0; -1.0; 0.0; 1.0; -1.0;;], [-1.0, -1.0, 0.0, 2.0, 1.0]]], "f_indices" => Any[1, 2, 3])

    linear_maps=get_linear_maps(pmap,reps["f_indices"])
    @test linear_maps==[Any[0//1, Rational{BigInt}[1]], Any[0//1, Rational{BigInt}[2]], Any[-1//1, Rational{BigInt}[3]]]

    polys=polyhedra_from_reps(reps)
    linear_regions=get_linear_regions(polys,linear_maps)
    level_set_component=get_level_set_component(linear_regions[Any[0//1, Rational{BigInt}[2]]]["polyhedra"][1],Any[0//1, Rational{BigInt}[2]],1.0)
    @test level_set_component==Any[BigFloat[0.5], BigFloat[0.5]]
    
    surface=get_surface_points(linear_regions[[-1//1, Rational{BigInt}[3]]]["polyhedra"][1],[-1//1, Rational{BigInt}[3]])
    @test surface==(Vector{Rational{BigInt}}[[1, 2]], Rational{BigInt}[2, 5])

    # one-dimensional rational map (0*T^0 + 0*T^3) / (0*T^1 + -1*T^2)
    pmap=TropicalNN.TropicalPuiseuxRational(TropicalPuiseuxPoly(Rational{BigInt}.([0,0]),[Rational{BigInt}.([3]),Rational{BigInt}.([0])],false),TropicalPuiseuxPoly(Rational{BigInt}.([0,-1]),[Rational{BigInt}.([1]),Rational{BigInt}.([2])],false))

    reps=formatted_reps(pmap)
    @test reps==Dict{String, Vector}("m_reps" => Vector{Array{Float64}}[[[0.0; 3.0; 0.0; 1.0; 1.0; -1.0;;], [0.0, 0.0, 0.0, 1.0, 2.0, 1.0]], [[-3.0; 0.0; 0.0; 1.0; 1.0; -1.0;;], [0.0, 0.0, 0.0, 1.0, 2.0, 1.0]], [[-3.0; 0.0; -1.0; 0.0; 1.0; -1.0;;], [0.0, 0.0, -1.0, 0.0, 2.0, 1.0]]], "f_indices" => Any[[1, 1], [2, 1], [2, 2]])

    linear_maps=get_linear_maps(pmap,reps["f_indices"])
    @test linear_maps==[Any[0//1, Rational{BigInt}[-1]], Any[0//1, Rational{BigInt}[2]], Any[1//1, Rational{BigInt}[1]]]

    # projecting representations

    pmap=TropicalPuiseuxPoly(Rational{BigInt}.([0,0.5,1,0.1]),[Rational{BigInt}.([0,1,1]),Rational{BigInt}.([0.5,0.5,0.5]),Rational{BigInt}.([1,0,0]),Rational{BigInt}.([0.5,0.5,1])],false)
    reps=formatted_reps(pmap,rot_matrix=[1 0 0;0 1 0;0 0 1])
    @test reps["m_reps"]==Any[Array{Float64}[[0.0 0.0; 0.5 -0.5; 0.5 -0.5; 1.0 -1.0; 1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0; 0.0 0.0; 0.0 0.0], [0.0, -0.5, -0.1, -1.0, 1.0, 1.2, 1.0, 1.0, 1.8, 1.0]], Array{Float64}[[-1.0 1.0; -0.5 0.5; -0.5 0.5; 0.0 0.0; 1.0 0.0; -1.0 0.0; 0.0 1.0; 0.0 -1.0; 0.0 0.0; 0.0 0.0], [1.0, 0.5, 0.9, 0.0, 1.0, 1.2, 1.0, 1.0, 1.8, 1.0]]]

    # Hoffman tests

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