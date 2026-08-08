"""
    TropicalNN

Tools for tropical polynomials, tropical rational functions, neural networks,
and linear-region analysis.
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
       AbstractNeuralNetworkLayer,
       AffineLayer,
       ActivationLayer,
       NeuralNetwork,
       input_dimension,
       output_dimension,
       relu,
       leaky_relu,
       maxout,
       identity_activation,
       signomial_const,
       signomial_zero,
       signomial_one,
       signomial_monomial,
       signomial_to_rational,
       rational_signomial_identity,
       rational_signomial_zero,
       rational_signomial_one,
       Signomial_const,
       Signomial_zero,
       Signomial_one,
       SignomialMonomial,
       RationalSignomial_identity,
       RationalSignomial_zero,
       RationalSignomial_one,
       exponents,
       coefficients,
       monomial_pairs,
       evaluate,
       quicksum,
       comp,
       monomial_count,
       single_to_trop,
       tropicalize,
       tropicalize_layers,
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
include("neural_network.jl")
include("linear_regions.jl")
include("tropicalize.jl")
include("util.jl")
include("monomial.jl")

include("hoffman.jl")

include("statistics.jl")
include("deprecated.jl")

end
