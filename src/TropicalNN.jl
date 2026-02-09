module TropicalNN

    using Oscar
    using Combinatorics
    using Distributions

    using Polyhedra
    using CDDLib
    using ColorSchemes
    using CairoMakie
    using Random

    using JuMP
    using GLPK
    using LinearAlgebra

    using Graphs
    using MetaGraphsNext

    import Base: string, +, *, /

    export
        TropicalPuiseuxPoly,
        TropicalPuiseuxRational,
        TropicalPuiseuxPoly_const,
        TropicalPuiseuxPoly_zero,
        TropicalPuiseuxPoly_one,
        TropicalPuiseuxMonomial,
        TropicalPuiseuxPoly_to_rational,
        TropicalPuiseuxRational_identity,
        TropicalPuiseuxRational_zero,
        TropicalPuiseuxRational_one,
        StandardizedTropicalPoly,
        StandardizedTropicalPoly_const,
        StandardizedTropicalPoly_zero,
        StandardizedTropicalPoly_one,
        StandardizedTropicalMonomial,
        standardize,
        destandardize,
        convert_denominator,
        eval_horner,
        eval_horner_univariate,
        eval_horner_multivariate,
        power_standardized,
        scalar_mult_standardized,
        string,
        eval,
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
        single_to_trop, 
        mlp_to_trop,
        mlp_to_trop_with_mul_with_quicksum,
        mlp_to_trop_with_quicksum, 
        random_mlp, 
        mlp_to_trop_with_dedup, 
        monomial_strong_elim, 
        mlp_to_trop_with_strong_elim, 
        mlp_to_trop_with_quicksum_with_strong_elim, 
        polyhedron, 
        enum_linear_regions,
        n_components, 
        components, 
        enum_linear_regions_rat,

        random_pmap,
        update_bounding_box,
        get_full_bounding_box,
        apply_linear_map,
        get_level_set_component,
        get_surface_points,
        get_linear_maps,
        get_linear_regions,
        bound_reps,
        project_reps,
        intersect_reps,
        m_reps,
        formatted_reps,
        polyhedra_from_reps,
        plotpoly,
        plotsurface,
        plotlevelset,
        plot_linear_regions,
        plot_linear_maps,

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
    include("mlp_to_trop.jl")
    include("mlp_to_trop_with_elim.jl")

    include("visualise.jl")

    include("hoffman.jl")

    include("statistics.jl")
end
