# API Reference

## Core types

```@docs
Signomial
RationalSignomial
AbstractNeuralNetworkLayer
AffineLayer
ActivationLayer
NeuralNetwork
Cell
LinearRegion
LinearRegions
```

## Construction

```@docs
signomial_const
signomial_zero
signomial_one
signomial_monomial
signomial_to_rational
rational_signomial_identity
rational_signomial_zero
rational_signomial_one
```

## Arithmetic

```@docs
evaluate
quicksum
comp
```

## Network conversion

```@docs
single_to_trop
tropicalize
tropicalize_layers
random_mlp
random_maxout_network
random_signomial
prune
```

## Neural networks

```@docs
relu
leaky_relu
maxout
identity_activation
input_dimension
output_dimension
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
