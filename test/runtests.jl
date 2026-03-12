using Test

@testset "TropicalNN.jl" begin

    include("./UnitTests/main.jl")

    include("./UnitTests/polynomial_algebra.jl")

    # TODO: StandardizedTropicalPoly functions not yet implemented
    # include("./UnitTests/standardized_poly.jl")

    include("./UnitTests/mlp_to_trop.jl")

    if Base.find_package("CairoMakie") !== nothing
        include("./UnitTests/visualise.jl")
    end

    include("./UnitTests/hoffman.jl")

    include("./UnitTests/statistics.jl")

    include("./UnitTests/linear_regions_highs.jl")

end