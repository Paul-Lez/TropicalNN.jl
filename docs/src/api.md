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
reduce
```

## Linear Regions

```@docs
LinearRegionsCalculationMode
OscarMode
HiGHSMode
polyhedron
get_matrix
get_vector
enum_linear_regions_general
linear_regions
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
