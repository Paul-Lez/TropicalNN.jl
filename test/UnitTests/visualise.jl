#using Test, TropicalNN, Oscar, GLMakie
using Test, TropicalNN, Oscar, CairoMakie

@testset "Visualise" begin

    # random_pmap
    pmap=random_pmap(2,4)
    @test typeof(pmap)<: TropicalPuiseuxPoly{Rational{BigInt}}
    @test nvars(pmap)==2
    @test monomial_count(pmap)==4

    fig,ax=plot_linear_regions(pmap)
    @test typeof(fig) <: Figure && typeof(ax) <: Axis
    fig,ax=plot_linear_regions(pmap,bounding_box=Dict(1 => [-1,1], 2 => [-1,1]))
    @test typeof(fig) <: Figure && typeof(ax) <: Axis

    fig,ax=plot_linear_maps(pmap)
    @test typeof(fig) <: Figure && typeof(ax) <: Axis3
    fig,ax=plot_linear_maps(pmap,bounding_box=Dict(1 => [-1,1], 2 => [-1,1]))
    @test typeof(fig) <: Figure && typeof(ax) <: Axis3

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
    @test level_set_component==Any[BigFloat[0.5]]
    
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

end