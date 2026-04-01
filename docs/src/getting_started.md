# Getting Started

## Full Pipeline Example

The following example converts a random 2-hidden-layer MLP to a tropical rational function,
enumerates its linear regions, and computes basic statistics.

```julia
using TropicalNN

# 1. Generate a random MLP with architecture [2, 4, 2, 1]
W, b, t = random_mlp([2, 4, 2, 1])

# 2. Convert to a tropical rational function
trop = mlp_to_trop(W, b, t)

# 3. Enumerate the linear regions of the first output
regions = enum_linear_regions_rat(trop[1])
println("Number of linear regions: ", length(regions))

# 4. Count monomials (expressivity measure)
println("Monomial count: ", monomial_count(trop[1]))
```

## Tropical Arithmetic

Tropical polynomials (signomials) support the standard arithmetic operations:

```julia
f = Signomial([1, 2, 3], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]])
g = Signomial([0, 4, -5], [[1//1, 7//1], [0//1, 1//1], [9//1, 1//1]])

h = f + g   # tropical sum (pointwise max)
p = f * g   # tropical product (sum of exponents, sum of coefficients)
```

## Performance Options

For larger networks, use `quicksum` and `strong_elim` to reduce monomial count:

```julia
trop_fast = mlp_to_trop(W, b, t; quicksum=true, strong_elim=true)
```
