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
├── linear_regions_calculation_general.jl # Linear region enumeration backends
├── statistics.jl          # Statistical analysis of regions
├── hoffman.jl             # Hoffman constant computation
└── visualise.jl           # Plotting utilities (CairoMakie)
```

## Key Types (tropical_poly_interface.jl)

```julia
abstract type AbstractSignomial{T} end

struct SignomialStatic{T, N} <: AbstractSignomial{T}
    coeff::Vector
    exp::Vector
end

struct SignomialMatrix{T} <: AbstractSignomial{T}
    exp::Matrix{T}
    coeff::Vector
end

struct RationalSignomial{P <: AbstractSignomial}
    num::P
    den::P
end
```

- `Signomial`: Constructor that chooses a static or matrix-backed polynomial representation
- `SignomialStatic` / `SignomialMatrix`: Concrete tropical polynomial representations
- `RationalSignomial`: Quotient of two tropical polynomials

## Core Functions by Module

### tropical_poly_interface.jl - Tropical Algebra
- `Signomial(coeffs, exps, sorted)` - Constructor
- `SignomialMonomial(coeff, exp)` - Single monomial
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
- `reduce(f)` - Remove redundant monomials (non-full-dim polyhedra)
- `mlp_to_trop_with_strong_elim()` - Convert with elimination
- `mlp_to_trop_with_quicksum_with_strong_elim()` - Combined optimization

### linear_regions_calculation_general.jl - Region Analysis
- `polyhedron(f, i, mode)` - Get backend region for i-th monomial
- `linear_regions(f; mode)` - Enumerate linear-region candidates of a polynomial
- `linear_regions(f; mode)` - Enumerate linear regions of a rational function
- `OscarMode()` / `HiGHSMode()` - Select exact Oscar or LP-backed HiGHS calculations
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
    ↓ linear_regions(...; mode)
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
