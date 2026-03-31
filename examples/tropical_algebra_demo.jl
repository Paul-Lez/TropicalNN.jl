# Demo: algebra of tropical Puiseux polynomials and rational functions
#
# We illustrate tropical addition (+), multiplication (*), and division (/)
# using two flavours of exponents:
#   - Rational{Int64} exponents  (exact arithmetic)
#   - Float64 exponents          (floating-point arithmetic)
#
# Recall the tropical semiring rules:
#   f ⊕ g  = max(f, g)   (tropical addition)
#   f ⊗ g  = f + g       (tropical multiplication)
#   f ⊘ g  = f - g       (tropical division, i.e. forming a rational function)

using TropicalNN
using Oscar

# ===========================================================================
# PART 1 — Rational exponents
# ===========================================================================

println("=" ^ 60)
println("PART 1 — Rational{Int64} exponents")
println("=" ^ 60)
println()

# --- Define polynomials ---
#
#   f = max(x, 2y, x + (1/2)y)        in two variables
#   g = max((1/2)x + y, 3y)

f = TropicalPuiseuxPoly(
    [0, 0, 0],
    [Rational{Int64}[1, 0], Rational{Int64}[0, 2], Rational{Int64}[1, 1//2]]
)

g = TropicalPuiseuxPoly(
    [0, 0],
    [Rational{Int64}[1//2, 1], Rational{Int64}[0, 3]]
)

println("f = ", f)
println("g = ", g)
println()

# --- Addition (tropical ⊕ = pointwise max) ---
#
#   f ⊕ g = max(x, 2y, x + (1/2)y, (1/2)x + y, 3y)

h_add = f + g
println("f ⊕ g = ", h_add)
println()

# --- Multiplication (tropical ⊗ = exponent-wise sum) ---
#
#   f ⊗ g = max over all pairs of monomials: exponents add, coefficients add

h_mul = f * g
println("f ⊗ g = ", h_mul)
println()

# --- Division: form a rational function f / g ---

q_rat = f / g
println("f ⊘ g = ", q_rat)
println()

# --- Evaluate at a point to verify ---
#
# In the tropical semiring a point x is a TropicalSemiringElem, but
# TropicalNN.evaluate() accepts a plain vector and does the semiring arithmetic
# internally.  We pass standard numbers and let Oscar convert them.

R = Oscar.tropical_semiring(max)
pt = [R(2), R(3)]   # x = 2, y = 3  (tropical numbers)

println("Evaluating at (x, y) = (2, 3):")
println("  f(2,3) = ", TropicalNN.evaluate(f, pt))
println("  g(2,3) = ", TropicalNN.evaluate(g, pt))
println("  (f⊕g)(2,3) = ", TropicalNN.evaluate(h_add, pt))
println("  (f⊗g)(2,3) = ", TropicalNN.evaluate(h_mul, pt))
println("  (f⊘g)(2,3) = ", TropicalNN.evaluate(q_rat, pt))
println()

# --- Rational function arithmetic ---
#
# Build two rational functions and form their product and sum.
#   p = f / g
#   r = (x + y) / max(x, y)

num2 = TropicalPuiseuxPoly(
    [0],
    [Rational{Int64}[1, 1]]
)
den2 = TropicalPuiseuxPoly(
    [0, 0],
    [Rational{Int64}[1, 0], Rational{Int64}[0, 1]]
)

p = f / g   # same as q_rat above
r = num2 / den2

println("p = ", p)
println("r = ", r)
println()

# Product of two rational functions: (f/g) * (num2/den2) = (f*num2) / (g*den2)
pr_product = p * r
println("p ⊗ r = ", pr_product)
println()

# Sum of two rational functions: (f/g) + (num2/den2) = (f*den2 + g*num2) / (g*den2)
pr_sum = p + r
println("p ⊕ r = ", pr_sum)
println()

# ===========================================================================
# PART 2 — Float64 exponents
# ===========================================================================

println("=" ^ 60)
println("PART 2 — Float64 exponents")
println("=" ^ 60)
println()

# --- Define polynomials ---
#
#   a = max(1.5x, 0.5y, x + 0.25y)
#   b = max(0.5x + 0.5y, 1.5y)
#
# All exponents are exact binary fractions, so floating-point arithmetic
# on them produces exact results (no rounding noise).
# Coefficients are still integers (the additive shift in max-plus algebra).

a = TropicalPuiseuxPoly(
    [0, 0, 0],
    [[1.5, 0.0], [0.0, 0.5], [1.0, 0.25]]
)

b = TropicalPuiseuxPoly(
    [0, 0],
    [[0.5, 0.5], [0.0, 1.5]]
)

println("a = ", a)
println("b = ", b)
println()

# --- Addition ---

ab_add = a + b
println("a ⊕ b = ", ab_add)
println()

# --- Multiplication ---

ab_mul = a * b
println("a ⊗ b = ", ab_mul)
println("  (", monomial_count(ab_mul), " monomials before any pruning)")
println()

# --- Division: rational function ---

q_float = a / b
println("a ⊘ b = ", q_float)
println()

# --- Evaluate at a point ---

pt_f = [R(1), R(1)]   # x = 1, y = 1

println("Evaluating at (x, y) = (1, 1):")
println("  a(1,1) = ", TropicalNN.evaluate(a, pt_f))
println("  b(1,1) = ", TropicalNN.evaluate(b, pt_f))
println("  (a⊕b)(1,1) = ", TropicalNN.evaluate(ab_add, pt_f))
println("  (a⊗b)(1,1) = ", TropicalNN.evaluate(ab_mul, pt_f))
println("  (a⊘b)(1,1) = ", TropicalNN.evaluate(q_float, pt_f))
