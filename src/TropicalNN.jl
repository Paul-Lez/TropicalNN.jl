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
       exponents,
       coefficients,
       monomial_pairs,
       evaluate,
       quicksum,
       comp,
       dedup_monomials,
       monomial_count,
       single_to_trop,
       tropicalize,
       mlp_to_trop,
       random_mlp,
       random_signomial,
       prune,
       OscarMode,
       HiGHSMode,
       linear_regions,
       Cell,
       LinearRegion,
       LinearRegions,
       hoffman_constant,
       upper_hoffman_constant,
       lower_hoffman_constant,
       exact_er,
       upper_er,
       interior_points,
       bounds,
       volumes,
       polyhedron_counts,
       edge_count,
       edge_lengths,
       edge_directions,
       edge_gradients,
       vertex_collection,
       vertex_count

include("signomial.jl")
include("rational_signomial.jl")
include("linear_regions.jl")
include("tropicalize.jl")
include("util.jl")
include("monomial.jl")

include("hoffman.jl")

include("statistics.jl")
include("deprecated.jl")

end
