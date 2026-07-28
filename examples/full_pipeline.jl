# Example: MLP to tropical rational function to linear regions
#
# Generate a ReLU MLP with architecture [2, 4, 1]. Convert it to
# a tropical rational function and compute its linear regions.

using Random
using TropicalNN

Random.seed!(2026)
weights, biases, thresholds = random_mlp([2, 4, 1])

println("Network architecture: [2, 4, 1]")
println("  Layer 1 weight matrix size: ", size(weights[1]))
println("  Layer 2 weight matrix size: ", size(weights[2]))
println()

f = only(mlp_to_trop(weights, biases, thresholds;
    quicksum = true, strong_elim = true))

println("Tropical rational function:")
println("  Numerator monomials:   ", monomial_count(f.num))
println("  Denominator monomials: ", monomial_count(f.den))
println()

# Compute regions with floating-point LP checks.
region_mode = HiGHSMode()
regions = linear_regions(f; mode = region_mode)

println("Linear regions found: ", length(regions))
println()

n_single_piece = count(region -> length(region) == 1, regions)
n_multiple_pieces = length(regions) - n_single_piece

println("Summary:")
println("  Regions with one convex piece:       ", n_single_piece)
println("  Regions with multiple convex pieces: ", n_multiple_pieces)
println()

n_show = min(3, length(regions))
println("First $n_show region(s):")
for i in 1:n_show
    region = regions[i]
    println("  Region $i: $(length(region)) convex piece(s)")
    for (piece_index, piece) in enumerate(region)
        A = get_matrix(piece; mode = region_mode)
        println("    Piece $piece_index: $(size(A, 1)) constraints")
    end
end
