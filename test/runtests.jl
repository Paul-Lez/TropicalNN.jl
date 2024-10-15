using TropicalNN
using Test
using Oscar

@testset "TropicalNN.jl" begin
    w, b, t = TropicalNN.random_mlp([3, 2, 1])
    trop = mlp_to_trop(w, b, t)[1]
    @show length(enum_linear_regions_rat(trop.num, trop.den))
    R = tropical_semiring(max)
    f = TropicalPuiseuxPoly([R(1), R(2), R(3)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
    # Write down the tropical polynomial 0*X^1*Y^7 + 4*X^0*Y^1 + (-5)*X^9*Y^1
    g = TropicalPuiseuxPoly([R(0), R(4), R(-5)], [[1//1, 7//1], [0//1, 1//1], [9//1, 1//1]], false) 
    @show f + g # outputs 
    @show f * g 
end
