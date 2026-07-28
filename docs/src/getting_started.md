# Getting Started

## Full pipeline

This example converts an MLP to a tropical rational function, computes its linear regions, and counts its stored monomials.

```julia
using TropicalNN

W, b, thresholds = random_mlp([2, 4, 2, 1])
q = mlp_to_trop(W, b, thresholds)
regions = linear_regions(q[1]; mode = HiGHSMode())

println("Linear regions: ", length(regions))
println("Stored monomials: ", monomial_count(q[1]))
```

## Tropical arithmetic

```julia
f = Signomial([1, 2, 3], [[1 // 1, 0 // 1], [0 // 1, 1 // 1], [1 // 1, 1 // 1]])
g = Signomial([0, 4, -5], [[1 // 1, 7 // 1], [0 // 1, 1 // 1], [9 // 1, 1 // 1]])

h = f + g # Pointwise maximum.
p = f * g # Ordinary sum of the represented functions.
```

## Control expression growth

You can use `quicksum` and `strong_elim` to accelerate computation and reduce intermediate expression size:

```julia
mode = HiGHSMode(threads = 4)
q_reduced = mlp_to_trop(
    W,
    b,
    thresholds;
    quicksum = true,
    strong_elim = true,
    elim_mode = mode,
)
pruned = prune(q_reduced[1]; mode = mode)
```
