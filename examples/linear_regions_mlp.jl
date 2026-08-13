# Example: linear regions of an MLP
#
# Construct a [2, 2, 1] ReLU MLP as a NeuralNetwork and convert it to a
# tropical rational function. Then use the layerwise implementation with
# HiGHS to compute its linear regions without forming the complete tropical
# expression as part of that computation.

using TropicalNN

W1 = Rational{BigInt}[0 1; 1 0]
b1 = Rational{BigInt}[1, 1]
W2 = Rational{BigInt}[1 -1]
b2 = Rational{BigInt}[0]

network = NeuralNetwork(
    AffineLayer(W1, b1),
    ActivationLayer(relu(), 2),
    AffineLayer(W2, b2)
)

f = only(tropicalize(network))
# f = (max(0, 1 + x_2)) / (max(0, 1 + x_1))

println("Tropical rational function for the network output:")
println("  Numerator monomials:   ", monomial_count(f.num))
println("  Denominator monomials: ", monomial_count(f.den))
println()

region_mode = HiGHSMode()
regions = linear_regions(network; mode = region_mode)

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
