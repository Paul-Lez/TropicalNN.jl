# Example: linear regions of an MLP
#
# Architecture [2, 2, 1]:
#   Layer 1: W = [0 1; 1 0], b = [1; 1], threshold = [0; 0]  (ReLU)
#   Layer 2: W = [1 -1],     b = [0],    threshold = [0]      (ReLU)
#
# The MLP is first converted to a tropical Puiseux rational function via
# mlp_to_trop, then its linear regions are enumerated using the HiGHS-based
# algorithm (enum_linear_regions_rat_highs).

using TropicalNN

# --- Define the MLP ----------------------------------------------------------

W1 = Rational{BigInt}[0 1; 1 0]
b1 = Rational{BigInt}[1, 1]
t1 = Rational{BigInt}[0, 0]

W2 = Rational{BigInt}[1 -1]
b2 = Rational{BigInt}[0]
t2 = Rational{BigInt}[0]

# --- Convert to tropical representation --------------------------------------

F = mlp_to_trop([W1, W2], [b1, b2], [t1, t2])
f = F[1]  # single output neuron

println("Tropical rational function for the network output:")
println("  Numerator monomials:   ", monomial_count(f.num))
println("  Denominator monomials: ", monomial_count(f.den))
println()

# --- Compute linear regions via HiGHS ----------------------------------------

regions = enum_linear_regions_rat_highs(f)

println("Number of linear regions: ", length(regions))
println()

for (i, region) in enumerate(regions)
    # A region is either a single (A, b) pair or a connected component
    # (array of (A, b) pairs sharing the same linear map).
    if region isa Tuple
        A, b = region
        println("Region $i:  {x : Ax ≤ b}")
        println("  A = ", A)
        println("  b = ", b)
    else
        println("Region $i:  connected component of $(length(region)) polyhedra")
        for (k, (A, b)) in enumerate(region)
            println("  Piece $k:  A = ", A, ",  b = ", b)
        end
    end
end
