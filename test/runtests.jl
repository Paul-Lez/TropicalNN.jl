using Test

@testset "TropicalNN.jl" begin

    include("./UnitTests/main.jl")

    include("./UnitTests/polynomial_algebra.jl")

    include("./UnitTests/mlp_to_trop.jl")

    # Visualise tests require CairoMakie and the visualise.jl src module to be loaded.
    # Commented out until visualise.jl is integrated into the module.
    # if Base.find_package("CairoMakie") !== nothing
    #     include("./UnitTests/visualise.jl")
    # end

    include("./UnitTests/hoffman.jl")

    include("./UnitTests/statistics.jl")

    include("./UnitTests/linear_regions_highs.jl")

end