# Example: tropical signomial algebra
#
# Tropical addition is pointwise maximum. Tropical multiplication is ordinary
# addition of function values. Tropical division is ordinary subtraction.

using TropicalNN

# Exact rational exponents
f = Signomial(
    [0, 0, 0],
    [Rational{Int}[1, 0], Rational{Int}[0, 2], Rational{Int}[1, 1 // 2]]
)
g = Signomial(
    [0, 0],
    [Rational{Int}[1 // 2, 1], Rational{Int}[0, 3]]
)

println("Exact exponent arithmetic:")
println("  f = ", f)
println("  g = ", g)
println("  f ⊕ g = ", f + g)
println("  f ⊗ g = ", f * g)
println("  f ⊘ g = ", f / g)
println("  f(2, 3) = ", evaluate(f, [2, 3]))
println("  (f ⊘ g)(2, 3) = ", evaluate(f / g, [2, 3]))

# Floating-point exponents
# Each exponent has a Float64 representation.
a = Signomial([0, 0, 0], [[1.5, 0.0], [0.0, 0.5], [1.0, 0.25]])
b = Signomial([0, 0], [[0.5, 0.5], [0.0, 1.5]])

println("\nFloating-point exponent arithmetic:")
println("  a = ", a)
println("  b = ", b)
println("  a ⊗ b = ", a * b)
println("  stored product terms = ", monomial_count(a * b))

# Composition
# outer(u, v) = max(u + v, 2u)
# inner_1(x, y) = max(x, y)
# inner_2(x, y) = max(2x, 0.5y)
inner_1 = Signomial([0, 0], [[1.0, 0.0], [0.0, 1.0]])
inner_2 = Signomial([0, 0], [[2.0, 0.0], [0.0, 0.5]])
outer = Signomial([0, 0], [[1.0, 1.0], [2.0, 0.0]])

composed = comp(outer, [inner_1, inner_2])
println("\nComposition:")
println("  outer(inner_1, inner_2) = ", composed)
