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
    # Visualise tests require CairoMakie and the visualise.jl src module to be loaded.
    # Omit them until visualise.jl is integrated into the module.
    unit_tests = [
        "main.jl",
        "polynomial_algebra.jl",
        "signomial.jl",
        "mlp_to_trop.jl",
        "hoffman.jl",
        "statistics.jl",
        "linear_regions_calculation_general.jl",
        "linear_regions_highs.jl",
        "exponentiation.jl",
        "printing.jl",
        "linearmap_matrices.jl",
        "tropical_number.jl"
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
