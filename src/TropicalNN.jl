"""
    TropicalNN

Tools for tropical Puiseux polynomials, tropical rational functions, and ReLU
MLP conversion. The main exported polynomial constructor is
`Signomial`, and the main exported rational-function type is
`RationalSignomial`.
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
import Oscar: convention

export
       convention,
# Concrete polynomial and rational-function types
       Signomial,
# Primary names (used throughout the module)
       RationalSignomial,
       OptimalTropicalRational,
       Signomial_const,
       Signomial_zero,
       Signomial_one,
       SignomialMonomial,
       signomial_to_rational,
       RationalSignomial_identity,
       RationalSignomial_zero,
       RationalSignomial_one,
# Internal accessor API
       get_exp,
       get_coeff,
       get_coeff_by_exp,
       eval_poly,
       eval_rational,
       exponents,
       coefficients,
       monomial_pairs,
       string,
       evaluate,
       +,
       /,
       *,
       ==,
       quicksum,
       add_with_quicksum,
       div_with_quicksum,
       comp,
       comp_with_quicksum,
       dedup_monomials,
       reduce, 
       monomial_count,
       nvars,
       single_to_trop,
       mlp_to_trop,
       random_mlp,
       reduce,
       LinearRegionsCalculationMode,
       OscarMode,
       HiGHSMode,
       polyhedron,
       get_matrix,
       get_vector,
       codimension_le_one,
       regions_intersect_codimension_le_one,
       linear_regions,
       linear_regions,
       n_components,
       components,
       LinearRegion,
       LinearRegions,
       random_pmap, linearmap_matrices,
       tilde_matrices,
       tilde_vectors,
       positive_component,
       surjectivity_test,
       exact_hoff,
       upper_hoff,
       lower_hoff,
       exact_er,
       upper_er, separate_components,
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

include("signomial.jl")
include("rational_signomial.jl")
include("tropical_number.jl")
include("linear_regions.jl")
include("mlp_to_trop.jl")
include("util.jl")
include("monomial.jl")

include("hoffman.jl")

include("statistics.jl")

end
