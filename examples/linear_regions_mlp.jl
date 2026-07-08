# Example: linear regions of an MLP
#
# Architecture [2, 2, 1]:
#   Layer 1: W = [0 1; 1 0], b = [1; 1]  (ReLU)
#   Layer 2: W = [1 -1],     b = [0]     (affine output)
#
# The MLP is first converted to a tropical Puiseux rational function via
# mlp_to_trop, then its linear regions are enumerated using HiGHSMode().

using TropicalNN

# --- Define the MLP ----------------------------------------------------------

W1 = Rational{BigInt}[0 1; 1 0]
b1 = Rational{BigInt}[1, 1]

W2 = Rational{BigInt}[1 -1]
b2 = Rational{BigInt}[0]

# --- Convert to tropical representation --------------------------------------

F = mlp_to_trop([W1, W2], [b1, b2])
f = F[1]  # single output neuron

println("Tropical rational function for the network output:")
println("  Numerator monomials:   ", monomial_count(f.num))
println("  Denominator monomials: ", monomial_count(f.den))
println()

# --- Compute linear regions via HiGHS ----------------------------------------

region_mode = HiGHSMode()
regions = enum_linear_regions_rat_general(f; mode = region_mode)

println("Number of linear regions: ", length(regions))
println()

for (i, region) in enumerate(regions)
    # Each region is a LinearRegion containing one or more backend region objects.
    # A single convex piece has length 1; disconnected pieces have length > 1.
    if length(region) == 1
        A = get_matrix(region[1]; mode = region_mode)
        b = get_vector(region[1]; mode = region_mode)
        println("Region $i:  {x : Ax ≤ b}")
        println("  A = ", A)
        println("  b = ", b)
    else
        println("Region $i:  connected component of $(length(region)) polyhedra")
        for (k, piece) in enumerate(region)
            A = get_matrix(piece; mode = region_mode)
            b = get_vector(piece; mode = region_mode)
            println("  Piece $k:  A = ", A, ",  b = ", b)
        end
    end
end
