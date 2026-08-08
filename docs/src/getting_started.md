# Getting Started

## Full pipeline

This example converts an MLP to a tropical rational function, computes its linear regions, and counts its stored monomials.

```julia
using TropicalNN

W, b, thresholds = random_mlp([2, 4, 2, 1])
q = tropicalize(W, b, thresholds)
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

You can use `quicksum` and `prune` to accelerate computation and reduce intermediate expression size:

```julia
mode = HiGHSMode(threads = 4)
q_reduced = tropicalize(
    W,
    b,
    thresholds;
    quicksum = true,
    prune = true,
    elim_mode = mode,
)
pruned = prune(q_reduced[1]; mode = mode)
```

## Create a neural network

Create one layer for each affine map and each activation. `NeuralNetwork`
applies the layers in the given order.

```julia
network = NeuralNetwork(
    AffineLayer([1 0; 0 1], [0, 0]),
    ActivationLayer(relu(Int), 2),
    AffineLayer([1 1], [0])
)
q = tropicalize(network)
layer_maps = tropicalize_layers(network)

input_dimension(network)  # 2
output_dimension(network) # 1
```

`tropicalize_layers` converts each layer separately. The result contains one
vector of rational signomials for each layer. `tropicalize` composes the layer
maps. It returns one vector for the complete network.

All layers in a network must use the same scalar type. The default type for an
activation is `Rational{BigInt}`. If an affine layer uses a different type,
give that type to the activation. For example, `relu(Int)` matches the integer
affine layers above. `maxout(Float32, 2)` creates a `Float32` activation that
takes the maximum of two inputs.

`ActivationLayer(relu(Int), 2)` creates two ReLU units. Each unit receives one
input. `ActivationLayer(maxout(Int, 2), 3)` creates three maxout units. Each
unit receives a separate block of two inputs.
