using JuliaFormatter
using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))

@testset "Formatting" begin
    @test format(ROOT; overwrite = false, verbose = true)
end
