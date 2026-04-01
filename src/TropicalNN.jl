"""
    TropicalNN

Julia library implementing tropical geometric tools for neural network analysis.
Converts ReLU MLPs to tropical rational functions, enabling analysis of linear regions,
expressivity measures, and Hoffman constants.

# Overview

- **Tropical arithmetic**: [`Signomial`](@ref) (max-plus polynomial), [`RationalSignomial`](@ref)
- **MLP conversion**: [`mlp_to_trop`](@ref) — converts any ReLU MLP to a tropical rational function
- **Linear regions**: [`enum_linear_regions_rat`](@ref) — enumerate the polyhedral linear regions
- **Expressivity**: [`monomial_count`](@ref), [`exact_hoff`](@ref) — Hoffman constant bounds
- **Statistics**: [`bounds`](@ref), [`volumes`](@ref), [`edge_count`](@ref)

# Quick Start

```julia
using TropicalNN

# Define tropical polynomials
f = Signomial([0, 0, 1], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]])
g = Signomial([0], [[0//1, 0//1]])

# Enumerate linear regions of f/g
regions = enum_linear_regions_rat(f / g)
println("Number of linear regions: ", length(regions))

# Convert a random MLP to a tropical rational function
W, b, t = random_mlp([3, 2, 2])
trop = mlp_to_trop(W, b, t)
```

See [arXiv:2405.20174](https://arxiv.org/abs/2405.20174) for the theoretical background.

Both `TropicalPuiseuxPoly`/`TropicalPuiseuxRational` (paper names) and
`Signomial`/`RationalSignomial` (module names) are exported and refer to the same types.
"""
module TropicalNN

    using Oscar
    import Combinatorics
    using Distributions

    using Polyhedra
    using CDDLib

    using JuMP
    using GLPK
    using HiGHS
    using LinearAlgebra

    using Graphs
    using MetaGraphsNext

    import Base: string, +, *, /

    export
        # Primary names (used throughout the module)
        Signomial,
        RationalSignomial,
        Signomial_const,
        Signomial_zero,
        Signomial_one,
        SignomialMonomial,
        signomial_to_rational,
        RationalSignomial_identity,
        RationalSignomial_zero,
        RationalSignomial_one,
        # Paper-consistent aliases — TropicalPuiseuxPoly / TropicalPuiseuxRational
        # are the names used in the associated publication; both resolve to the same types.
        TropicalPuiseuxPoly,
        TropicalPuiseuxRational,
        TropicalPuiseuxPoly_const,
        TropicalPuiseuxPoly_zero,
        TropicalPuiseuxPoly_one,
        TropicalPuiseuxMonomial,
        TropicalPuiseuxRational_identity,
        TropicalPuiseuxRational_zero,
        TropicalPuiseuxRational_one,
        # TODO: StandardizedTropicalPoly functions not yet implemented
        # StandardizedTropicalPoly,
        # StandardizedTropicalPoly_const,
        # StandardizedTropicalPoly_zero,
        # StandardizedTropicalPoly_one,
        # StandardizedTropicalMonomial,
        # standardize,
        # destandardize,
        # convert_denominator,
        # eval_horner,
        # eval_horner_univariate,
        # eval_horner_multivariate,
        # power_standardized,
        # scalar_mult_standardized,
        string,
        evaluate,
        +,
        /,
        *,
        ==,
        quicksum,
        mul_with_quicksum,
        add_with_quicksum,
        div_with_quicksum,
        comp,
        comp_with_quicksum,
        dedup_monomials,
        monomial_count,
        nvars,
        single_to_trop,
        mlp_to_trop,
        random_mlp,
        monomial_strong_elim,
        polyhedron,
        enum_linear_regions,
        n_components,
        components,
        LinearRegion,
        LinearRegions,
        enum_linear_regions_rat,
        polyhedron_highs,
        enum_linear_regions_highs,
        enum_linear_regions_rat_highs,

        random_pmap,

        linearmap_matrices,
        tilde_matrices,
        tilde_vectors,
        positive_component,
        surjectivity_test,
        exact_hoff,
        upper_hoff,
        lower_hoff,
        exact_er,
        upper_er,

        separate_components,
        map_statistic,
        interior_points,
        bounds,
        volumes,
        polyhedron_counts,
        get_graph,
        edge_count,
        edge_lengths,
        edge_directions,
        edge_gradients,
        vertex_collection,
        vertex_count

    include("rat_maps.jl")
    include("linear_regions.jl")
    include("linear_regions_highs.jl")
    include("mlp_to_trop.jl")
    include("mlp_to_trop_with_elim.jl")

    include("hoffman.jl")

    include("statistics.jl")

    # ---------------------------------------------------------------------------
    # Paper-consistent type and function aliases.
    # The publication uses TropicalPuiseuxPoly / TropicalPuiseuxRational;
    # both names resolve to exactly the same types and functions.
    # ---------------------------------------------------------------------------
    const TropicalPuiseuxPoly        = Signomial
    const TropicalPuiseuxRational    = RationalSignomial
    const TropicalPuiseuxPoly_const  = Signomial_const
    const TropicalPuiseuxPoly_zero   = Signomial_zero
    const TropicalPuiseuxPoly_one    = Signomial_one
    const TropicalPuiseuxMonomial    = SignomialMonomial
    const TropicalPuiseuxRational_identity = RationalSignomial_identity
    const TropicalPuiseuxRational_zero     = RationalSignomial_zero
    const TropicalPuiseuxRational_one      = RationalSignomial_one
end
