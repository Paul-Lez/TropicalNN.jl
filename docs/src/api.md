# API Reference

## Core types

```@docs
Signomial
RationalSignomial
Cell
LinearRegion
LinearRegions
```

## Construction

```@docs
Signomial_const
Signomial_zero
Signomial_one
SignomialMonomial
signomial_to_rational
RationalSignomial_identity
RationalSignomial_zero
RationalSignomial_one
```

## Arithmetic

```@docs
evaluate
quicksum
```

## MLP conversion

```@docs
single_to_trop
tropicalize
random_mlp
random_signomial
prune
```

## Linear regions

```@docs
OscarMode
HiGHSMode
linear_regions
```

## Statistics

```@docs
interior_points
bounds
volumes
polyhedron_counts
edge_count
edge_lengths
edge_directions
edge_gradients
vertex_collection
vertex_count
```

## Hoffman constants

```@docs
hoffman_constant
upper_hoffman_constant
lower_hoffman_constant
exact_er
upper_er
```
