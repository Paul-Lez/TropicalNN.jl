# TropicalNN

[![Build Status](https://github.com/Paul-Lez/TropicalNN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Paul-Lez/TropicalNN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://paul-lez.github.io/TropicalNN.jl/dev/)

## Installation

TropicalNN requires Julia 1.12. In Julia package mode, run:

```julia-repl
pkg> add https://github.com/Paul-Lez/TropicalNN.jl
```

Or run:

```julia
using Pkg
Pkg.add(url = "https://github.com/Paul-Lez/TropicalNN.jl")
```

## Content

TropicalNN provides symbolic tropical-geometry tools for neural networks.

### Tropical expressions of neural networks

A fully connected ReLU multilayer perceptron (MLP) can be expressed as a tropical rational function with real exponents [1]. TropicalNN can construct and manipulate these expressions.

### Tropical measures of neural-network expressivity

TropicalNN can compute the linear regions of a neural network, analyze their geometry, and count the monomials in its stored tropical expression.

## Quick start

This example converts an MLP to a tropical rational function and computes the linear regions of its first output:

```julia
using TropicalNN

W, b, thresholds = random_mlp([2, 4, 2, 1])
q = mlp_to_trop(W, b, thresholds)
regions = linear_regions(q[1]; mode = HiGHSMode())

println("Linear regions: ", length(regions))
println("Stored monomials: ", monomial_count(q[1]))
```

See [`examples/full_pipeline.jl`](examples/full_pipeline.jl) for a complete example.

## Tropical expressions

```julia
using TropicalNN

f = Signomial([1, 2, 3], [[1 // 1, 0 // 1], [0 // 1, 1 // 1], [1 // 1, 1 // 1]])
g = Signomial([0, 4, -5], [[1 // 1, 7 // 1], [0 // 1, 1 // 1], [9 // 1, 1 // 1]])

@show f + g # Pointwise maximum.
@show f * g # Ordinary sum of the represented functions.
```

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
```

## Citation

If you use TropicalNN in research, cite:

```bibtex
@article{lezeau2024tropical,
  title = {Tropical Expressivity of Neural Networks},
  author = {Lezeau, Paul and Walker, Thomas and Cao, Yueqi and Bhatia, Shiv and Monod, Anthea},
  journal = {arXiv preprint arXiv:2405.20174},
  year = {2024}
}
```

## Documentation

See the [development documentation](https://paul-lez.github.io/TropicalNN.jl/dev/).

## Reference

[1] Liwen Zhang, Gregory Naitzat, and Lek-Heng Lim, [*Tropical Geometry of Deep Neural Networks*](https://arxiv.org/abs/1805.07091).
