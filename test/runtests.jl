using Test

if "format" in ARGS
    @testset verbose = true "TropicalNN.jl formatting" begin
        @testset "formatting.jl" begin
            println(stderr, "Running test/formatting.jl")
            flush(stderr)
            include("formatting.jl")
        end
    end
else
    unit_tests = [
        "cells.jl",
        "polynomial_algebra.jl",
        "signomial.jl",
        "neural_network.jl",
        "tropicalize.jl",
        "hoffman.jl",
        "statistics.jl",
        "linear_regions_calculation_general.jl",
        "linear_regions_highs.jl",
        "exponentiation.jl",
        "printing.jl",
        "linearmap_matrices.jl"
    ]

    @testset verbose = true "TropicalNN.jl" begin
        for file in unit_tests
            path = joinpath("UnitTests", file)
            println(stderr, "Running test/$path")
            flush(stderr)

            @testset "UnitTests/$file" begin
                include(path)
            end
        end
    end
end
