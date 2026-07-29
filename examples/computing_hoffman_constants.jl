# Example: Hoffman constants and bounds
#
# hoffman_constant uses the PVZ algorithm by default. With brute_force=true, it
# tests every nonempty row subset instead. Both modes solve floating-point LPs.
# upper_hoffman_constant uses singular values, and lower_hoffman_constant
# samples row subsets.

using Random
using TropicalNN

A = [1.0 0.0; 0.0 1.0; -1.0 -1.0]

println("Matrix:")
println("  PVZ value:        ", hoffman_constant(A))
println("  enumerated value: ", hoffman_constant(A; brute_force = true))
println("  upper bound:      ", upper_hoffman_constant(A))

Random.seed!(2026)
println("  sampled bound:    ", lower_hoffman_constant(A, 5))

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
println("  PVZ value:        ", hoffman_constant(f))
println("  enumerated value: ", hoffman_constant(f; brute_force = true))
println("  upper bound:      ", upper_hoffman_constant(f))

println("\nTropical rational function:")
println("  PVZ value:        ", hoffman_constant(q))
println("  enumerated value: ", hoffman_constant(q; brute_force = true))
println("  upper bound:      ", upper_hoffman_constant(q))

# exact_er returns an effective-radius bound about the origin.
println("\nEffective-radius bounds:")
println("  polynomial:       ", exact_er(f))
println("  rational function: ", exact_er(q))
