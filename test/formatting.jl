using JuliaFormatter
using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FORMAT_PATHS = [
    "src",
    "ext",
    "test",
    "examples",
    joinpath("docs", "make.jl"),
    joinpath("docs", "src")
]

@testset "Formatting" begin
    @test format(joinpath.(ROOT, FORMAT_PATHS); overwrite = false, verbose = true)
end
