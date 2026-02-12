# CLAUDE.md - TropicalNN.jl

## Overview

Julia library implementing tropical geometric tools for neural network analysis. Converts ReLU MLPs to tropical rational functions, enabling analysis of linear regions, expressivity measures, and Hoffman constants.

## Project Structure

```
src/
├── TropicalNN.jl          # Module definition and exports
├── rat_maps.jl            # Core tropical algebra types and operations
├── mlp_to_trop.jl         # MLP to tropical conversion
├── mlp_to_trop_with_elim.jl # Conversion with monomial elimination
├── linear_regions.jl      # Linear region enumeration
├── statistics.jl          # Statistical analysis of regions
├── hoffman.jl             # Hoffman constant computation
└── visualise.jl           # Plotting utilities (CairoMakie)
```

## Key Types (rat_maps.jl)

```julia
struct TropicalPuiseuxPoly{T}
    coeff::Dict           # Coefficient dictionary (exponent vector → coefficient)
    exp::Vector{Vector{T}}  # Exponent vectors (lexicographically sorted)
end

struct TropicalPuiseuxRational{T}
    num::TropicalPuiseuxPoly{T}
    den::TropicalPuiseuxPoly{T}
end
```

- `TropicalPuiseuxPoly`: Tropical polynomial with rational exponents
- `TropicalPuiseuxRational`: Quotient of two tropical polynomials

## Core Functions by Module

### rat_maps.jl - Tropical Algebra
- `TropicalPuiseuxPoly(coeffs, exps, sorted)` - Constructor
- `TropicalPuiseuxMonomial(coeff, exp)` - Single monomial
- `+`, `*`, `/` - Arithmetic operators
- `eval(f, x)` - Evaluate at point
- `comp(f, g)` - Function composition
- `quicksum()`, `*_with_quicksum()` - Fast variants trading accuracy for speed

### mlp_to_trop.jl - MLP Conversion
- `random_mlp(dims)` - Generate random MLP (He initialization)
- `single_to_trop(A, b, t)` - Convert single layer max(Ax+b, t)
- `mlp_to_trop(W, b, t)` - Full MLP to tropical rational
- `mlp_to_trop_with_quicksum()` - Optimized version

### mlp_to_trop_with_elim.jl - Optimization
- `monomial_strong_elim(f)` - Remove redundant monomials (non-full-dim polyhedra)
- `mlp_to_trop_with_strong_elim()` - Convert with elimination
- `mlp_to_trop_with_quicksum_with_strong_elim()` - Combined optimization

### linear_regions.jl - Region Analysis
- `polyhedron(f, i)` - Get polyhedron for i-th monomial
- `enum_linear_regions(f)` - Enumerate linear regions of polynomial
- `enum_linear_regions_rat(f)` - For rational functions
- `components(V, D)` / `n_components(V, D)` - Connected components

### statistics.jl - Statistical Analysis
- `separate_components()` - Partition regions by connectivity
- `interior_points()` - Find interior points of polyhedra
- `bounds()`, `volumes()` - Geometric properties
- `get_graph()` - Build graph of linear regions

### hoffman.jl - Hoffman Constants
- `surjectivity_test(A)` - LP-based A-surjectivity test
- `exact_hoff(A)` - Exact Hoffman constant (brute force)
- `upper_hoff(A)` / `lower_hoff(A, samples)` - Bounds
- `linearmap_matrices(f)` - Extract matrix representations

### visualise.jl - Plotting
- `plot_linear_regions(f)` - Visualize polyhedra (1D/2D/3D)
- `plot_linear_maps(f)` - Visualize linear map actions

## Data Flow

```
MLP (weights, biases, thresholds)
    ↓ mlp_to_trop()
Tropical Puiseux Rational
    ↓ enum_linear_regions_rat()
Linear Regions (Polyhedra)
    ↓
Statistics / Visualization
```

## Key Dependencies

- **Oscar.jl**: Tropical semiring, polyhedra
- **Polyhedra.jl + CDDLib.jl**: Polyhedron manipulation
- **JuMP + GLPK**: Linear programming
- **Graphs + MetaGraphsNext**: Graph analysis
- **CairoMakie**: Visualization

## Common Patterns

1. **Quicksum variants**: Functions with `_with_quicksum` trade accuracy for speed
2. **Monomial elimination**: `_with_strong_elim` removes redundant monomials
3. **Lexicographic ordering**: Exponents always sorted for efficient lookup
4. **Type parameter T**: Supports `Float64`, `Rational{Int64}`, `Rational{BigInt}`

## Build/Test Commands

```bash
julia +1.10 --project -e 'using Pkg; Pkg.test()'
```

To run a testing script:
```bash
julia +1.10 --project=. path_to_script
```
