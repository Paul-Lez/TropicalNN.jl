"""
    TropicalNN

Tools for tropical polynomials, tropical rational functions, and MLP
conversion.
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

import Distributed

export
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
       get_exp,
       get_coeff,
       get_coeff_by_exp,
       exponents,
       coefficients,
       monomial_pairs,
       evaluate,
       quicksum,
       comp,
       comp_with_quicksum,
       dedup_monomials,
       monomial_count,
       single_to_trop,
       mlp_to_trop,
       random_mlp,
       random_signomial,
       prune,
       LinearRegionsCalculationMode,
       OscarMode,
       HiGHSMode,
       polyhedron,
       get_matrix,
       get_vector,
       linear_regions,
       components,
       LinearRegion,
       LinearRegions,
       linearmap_matrices,
       tilde_matrices,
       tilde_vectors,
       positive_component,
       surjectivity_test,
       exact_hoff,
       pvz_hoff,
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
include("linear_regions.jl")
include("mlp_to_trop.jl")
include("util.jl")
include("monomial.jl")

include("hoffman.jl")

include("statistics.jl")

end
