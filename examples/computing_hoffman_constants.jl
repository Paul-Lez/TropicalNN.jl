# Example: Hoffman constants and bounds
#
# exact_hoff tests every nonempty row subset. pvz_hoff omits subsets that an
# earlier test classified. Both functions solve floating-point LPs. upper_hoff
# uses singular values. lower_hoff samples row subsets.

using Random
using TropicalNN

A = [1.0 0.0; 0.0 1.0; -1.0 -1.0]

println("Matrix:")
println("  enumerated value: ", exact_hoff(A))
println("  PVZ value:        ", pvz_hoff(A))
println("  upper bound:      ", upper_hoff(A))

Random.seed!(2026)
println("  sampled bound:    ", lower_hoff(A, 5))

f = Signomial(
    [0, 0, 0],
    [[0 // 1, 0 // 1], [1 // 1, 0 // 1], [0 // 1, 1 // 1]]
)
g = Signomial(
    [0, 0],
    [[1 // 1, 0 // 1], [0 // 1, 1 // 1]]
)
q = f / g

println("\nTropical polynomial:")
println("  enumerated value: ", exact_hoff(f))
println("  upper bound:      ", upper_hoff(f))

println("\nTropical rational function:")
println("  enumerated value: ", exact_hoff(q))
println("  upper bound:      ", upper_hoff(q))

# exact_er returns an effective-radius bound about the origin.
println("\nEffective-radius bounds:")
println("  polynomial:       ", exact_er(f))
println("  rational function: ", exact_er(q))
