# Example: pruning a TropicalPuiseux polynomial
#
# We define the tropical Puiseux polynomial
#
#   f = max(x, y, 2x+2y, x+y, 2x, 2y)
#
# in 2 variables.  In the max-plus convention each monomial with exponent
# vector α = [α₁, α₂] and coefficient c represents the linear function
# c + α₁·x₁ + α₂·x₂, so the polynomial above has six monomials, all with
# coefficient 0:
#
#   exponent [1,0]  →  x₁
#   exponent [0,1]  →  x₂
#   exponent [2,2]  →  2x₁ + 2x₂
#   exponent [1,1]  →  x₁ + x₂
#   exponent [2,0]  →  2x₁
#   exponent [0,2]  →  2x₂
#
# The pruning mechanism (monomial_strong_elim) removes redundant monomials —
# those whose region of dominance is not full-dimensional — without changing
# the function.

using TropicalNN

# --- Define the polynomial ---------------------------------------------------

exps   = [[1, 0], [0, 1], [2, 2], [1, 1], [2, 0], [0, 2]]
coeffs = [0, 0, 0, 0, 0, 0]

f = TropicalPuiseuxPoly(coeffs, exps)

println("Original polynomial:")
println("  Number of monomials: ", monomial_count(f))
println("  Exponents (sorted):  ", f.exp)

# --- Run the pruning mechanism -----------------------------------------------

f_pruned = monomial_strong_elim(f)

println("\nPruned polynomial:")
println("  Number of monomials: ", monomial_count(f_pruned))
println("  Exponents (sorted):  ", f_pruned.exp)
