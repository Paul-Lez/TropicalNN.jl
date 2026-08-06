# Example: linear regions of an MLP
#
# Convert a [2, 2, 1] ReLU MLP to a tropical rational function. Then use
# HiGHS to compute its linear regions.

using TropicalNN

W1 = Rational{BigInt}[0 1; 1 0]
b1 = Rational{BigInt}[1, 1]
W2 = Rational{BigInt}[1 -1]
b2 = Rational{BigInt}[0]

f = only(tropicalize([W1, W2], [b1, b2]))

println("Tropical rational function for the network output:")
println("  Numerator monomials:   ", monomial_count(f.num))
println("  Denominator monomials: ", monomial_count(f.den))
println()

region_mode = HiGHSMode()
regions = linear_regions(f; mode = region_mode)

println("Number of linear regions: ", length(regions))
println()

for (i, region) in enumerate(regions)
    println("Region $i has $(length(region)) convex piece(s).")
    for (piece_index, piece) in enumerate(region)
        A = TropicalNN.get_matrix(piece; mode = region_mode)
        b = TropicalNN.get_vector(piece; mode = region_mode)
        println("  Piece $piece_index: {x : A * x ≤ b}")
        println("    A = ", A)
        println("    b = ", b)
    end
end
