# TropicalNN

[![Build Status](https://github.com/Paul-Lez/TropicalNN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Paul-Lez/TropicalNN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://paul-lez.github.io/TropicalNN.jl/dev/)

## Installation

This package requires Julia 1.10. To install, open the Julia REPL, press `]` to enter Pkg mode, and run:

```
pkg> add https://github.com/Paul-Lez/TropicalNN.jl
```

Or from the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/Paul-Lez/TropicalNN.jl")
```

## Content
This library implements some basic symbolic tropical geometric tools for computing with neural networks in Julia.

### Tropical Expressions of Neural Networks
Every fully connected multilayer perceptron (MLP) with ReLU activation can be written in terms of tropical algebra [1]. More precisely, any such MLP can be expressed as a *tropical rational function*, if one allows the exponents appearing in such expressions to be non-integral (e.g. rational or real numbers). We provide tools for symbolically manipulating such objects, and for finding the tropical expression of arbitrary MLPs with ReLU activation. 

### Tropical Measures of Expressivity of Neural Networks
There are various ways of measuring how *complicated* the function represented by a given neural network is. A large body of litterature has formed around using the number of *linear regions* of a neural network as such a measure. This library provides tools for computing this quantity, analysing the geometry of these linear regions, and studying another algebraic measure of expressivity, namely the number of irredundant monomials that appear in the tropical expression of a neural network. 

## Quick Start

The following snippet walks through the complete MLP → tropical → regions pipeline;
for more details see [`examples/full_pipeline.jl`](examples/full_pipeline.jl).

```julia
using TropicalNN

# 1. Generate a random 2-hidden-layer MLP with architecture [2, 4, 2, 1]
W, b, t = random_mlp([2, 4, 2, 1])

# 2. Convert to a tropical rational function
trop = mlp_to_trop(W, b, t)

# 3. Enumerate the linear regions of the first output
regions = enum_linear_regions_rat(trop[1])
println("Number of linear regions: ", length(regions))

# 4. Count monomials (expressivity measure)
println("Monomial count: ", monomial_count(trop[1]))
```

## Examples of usage: 

### Manipulating Tropical Expressions
```julia
using TropicalNN

# Write down the tropical polynomial f = max(1+x, 2+y, 3+x+y)
f = Signomial([1, 2, 3], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]])
# Write down the tropical polynomial g = max(0+x+7y, 4+y, -5+9x+y)
g = Signomial([0, 4, -5], [[1//1, 7//1], [0//1, 1//1], [9//1, 1//1]])
@show f + g  # tropical sum (max of all monomials)
@show f * g  # tropical product (sum of exponents, sum of coefficients)
```

### Computing Tropical Expressions of MLPs
```julia
using TropicalNN

# Generate a random neural network: returns (W, b, t) — weight matrices, biases, thresholds
W, b, t = random_mlp([3, 2, 2])
# Compute a tropical rational expression of this network
trop1 = mlp_to_trop(W, b, t)
# Compute a reduced tropical expression (faster, fewer monomials)
trop2 = mlp_to_trop(W, b, t; quicksum=true, strong_elim=true)
```

### Computing Linear Regions of Tropical Rational Functions
```julia
using TropicalNN

f = Signomial([1, 2, 3], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]])
g = Signomial([0, 4, -5], [[1//1, 7//1], [0//1, 1//1], [9//1, 1//1]])
# Compute the linear regions of f/g as a LinearRegions object
linear_regions = enum_linear_regions_rat(f / g)
@show length(linear_regions)
```

If this code was useful, please consider citing [this paper](https://arxiv.org/abs/2405.20174), using
```
@article{lezeau2024tropical,
  title={Tropical Expressivity of Neural Networks},
  author={Lezeau, Paul and Walker, Thomas and Cao, Yueqi and Bhatia, Shiv and Monod, Anthea},
  journal={arXiv preprint arXiv:2405.20174},
  year={2024}
}
```

## Documentation
Full API documentation is available at https://paul-lez.github.io/TropicalNN.jl/dev/

## References
[1] [*Tropical Geometry of Deep Neural Networks*](https://arxiv.org/pdf/1805.07091), Liwen Zhang, Gregory Naitzat and Lek-Heng Lim.