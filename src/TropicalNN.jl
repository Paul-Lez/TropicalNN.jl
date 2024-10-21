module TropicalNN

    using Oscar
    using Combinatorics
    using Distributions

    using Polyhedra
    using CDDLib
    using ColorSchemes
    # using LinearAlgebra
    using GLMakie

    import Base: string, +, *, /

    export 
        TropicalPuiseuxPoly,
        TropicalPuiseuxPoly_const,
        TropicalPuiseuxPoly_zero, 
        TropicalPuiseuxPoly_one,
        TropicalPuiseuxMonomial,
        TropicalPuiseuxPoly_to_rational,
        TropicalPuiseuxRational_identity,
        TropicalPuiseuxRational_zero,
        TropicalPuiseuxRational_one,
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
        get_level_set,
        get_surface_points,
        get_linear_maps,
        get_linear_regions,
        
        pmap_reps,
        bound_reps,
        project_reps,
        intersect_reps,
        m_reps,
        polyhedra_from_reps,

        plotpoly,
        plotsurface,
        plotlevelset,
        plot_linear_regions,
        plot_linear_maps


    include("rat_maps.jl")
    include("linear_regions.jl")
    include("mlp_to_trop.jl")
    include("mlp_to_trop_with_elim.jl")

    include("vis_utilities.jl")
    include("vis_representations.jl")
    include("vis_plotting.jl")
end
