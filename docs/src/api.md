# API Reference

## Core Types

```@docs
Signomial
RationalSignomial
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

## MLP Conversion

```@docs
mlp_to_trop
single_to_trop
random_mlp
random_pmap
monomial_strong_elim
```

## Linear Regions

```@docs
polyhedron
enum_linear_regions
enum_linear_regions_rat
polyhedron_highs
enum_linear_regions_highs
enum_linear_regions_rat_highs
components
```

## Statistics

```@docs
map_statistic
separate_components
interior_points
bounds
volumes
polyhedron_counts
get_graph
edge_count
edge_lengths
edge_directions
edge_gradients
vertex_collection
vertex_count
```

## Hoffman Constants

```@docs
surjectivity_test
exact_hoff
upper_hoff
lower_hoff
exact_er
upper_er
linearmap_matrices
tilde_matrices
tilde_vectors
positive_component
```

## Type Aliases

The following names are aliases for the primary types and constructors above,
using the terminology of the companion paper (arXiv:2405.20174).

The following names are exported as aliases pointing to the same objects as the primary names above:

| Alias | Primary name |
|---|---|
| `TropicalPuiseuxPoly` | [`Signomial`](@ref) |
| `TropicalPuiseuxRational` | [`RationalSignomial`](@ref) |
| `TropicalPuiseuxPoly_const` | [`Signomial_const`](@ref) |
| `TropicalPuiseuxPoly_zero` | [`Signomial_zero`](@ref) |
| `TropicalPuiseuxPoly_one` | [`Signomial_one`](@ref) |
| `TropicalPuiseuxMonomial` | [`SignomialMonomial`](@ref) |
| `TropicalPuiseuxRational_identity` | [`RationalSignomial_identity`](@ref) |
| `TropicalPuiseuxRational_zero` | [`RationalSignomial_zero`](@ref) |
| `TropicalPuiseuxRational_one` | [`RationalSignomial_one`](@ref) |
