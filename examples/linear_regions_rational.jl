# Example: linear regions of a tropical Puiseux rational function
#
# Compute the linear regions of
# q(x, y) = max(x, y, x + y) - max(x, x + 2y).

using TropicalNN

num_exps = [[1, 0], [0, 1], [1, 1]]
num_coeffs = [0, 0, 0]
f = Signomial(num_coeffs, num_exps)

den_exps = [[1, 0], [1, 2]]
den_coeffs = [0, 0]
g = Signomial(den_coeffs, den_exps)

q = f / g
region_mode = HiGHSMode()
regions = linear_regions(q; mode = region_mode)

println("Tropical rational function:  max(x, y, x+y) - max(x, x+2y)")
println("Number of linear regions: ", length(regions))
println()

for (i, region) in enumerate(regions)
    println("Region $i has $(length(region)) convex piece(s).")
    for (piece_index, piece) in enumerate(region)
        A = get_matrix(piece; mode = region_mode)
        b = get_vector(piece; mode = region_mode)
        println("  Piece $piece_index: {x : A * x ≤ b}")
        println("    A = ", A)
        println("    b = ", b)
    end
end
