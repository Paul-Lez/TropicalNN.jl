# TropicalNN.jl

**TropicalNN.jl** is a Julia library that converts ReLU multilayer perceptrons (MLPs) into
*tropical Puiseux rational functions*, enabling exact symbolic analysis of their linear regions,
expressivity, and Hoffman constants.

For the mathematical background, see the companion paper:
> Paul Lezeau, Thomas Walker, Yueqi Cao, Shiv Bhatia, Anthea Monod.
> *Tropical Expressivity of Neural Networks.* arXiv:2405.20174, 2024.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Paul-Lez/TropicalNN.jl")
```

## Navigation

- [Getting Started](@ref "Getting Started") — end-to-end pipeline example
- [API Reference](@ref "API Reference") — full function reference
