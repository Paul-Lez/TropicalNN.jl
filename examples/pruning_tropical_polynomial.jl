# Example: pruning a tropical polynomial
#
# Remove the redundant x + y term from
# f(x, y) = max(x, y, 2x + 2y, x + y, 2x, 2y).

using TropicalNN

exps = [[1, 0], [0, 1], [2, 2], [1, 1], [2, 0], [0, 2]]
coeffs = [0, 0, 0, 0, 0, 0]

f = Signomial(coeffs, exps)

println("Original polynomial:")
println("  Number of monomials: ", monomial_count(f))
println("  Exponents (sorted):  ", exponents(f))

f_pruned = prune(f)

println("\nPruned polynomial:")
println("  Number of monomials: ", monomial_count(f_pruned))
println("  Exponents (sorted):  ", exponents(f_pruned))
