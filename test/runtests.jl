using TropicalNN
using Test

@testset "TropicalNN.jl" begin
    w, b, t = TropicalNN.random_mlp([3, 2, 1])
    trop = TropicalNN.mlp_to_trop(w, b, t)[1]
    @show length(TropicalNN.enum_linear_regions_rat(trop.num, trop.den))
end
