using Test

@testset "TropicalNN.jl" begin

    include("./UnitTests/main.jl")

    include("./UnitTests/polynomial_algebra.jl")

    include("./UnitTests/standardized_poly.jl")

    include("./UnitTests/mlp_to_trop.jl")

    include("./UnitTests/visualise.jl")

    include("./UnitTests/hoffman.jl")

    include("./UnitTests/statistics.jl")

end