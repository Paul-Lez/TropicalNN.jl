module TropicalNN

    using Oscar
    using Combinatorics
    using Distributions

    include("rat_maps.jl")
    include("linear_regions.jl")
    include("mlp_to_trop.jl")
    include("mlp_to_trop_with_elim.jl")
end
