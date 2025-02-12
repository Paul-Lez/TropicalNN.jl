using Test

@testset "TropicalNN.jl" begin
   
    include("./UnitTests/main.jl")

    include("./UnitTests/visualise.jl")

    include("./UnitTests/hoffman.jl")

    include("./UnitTests/statistics.jl")

end