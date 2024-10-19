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

end
