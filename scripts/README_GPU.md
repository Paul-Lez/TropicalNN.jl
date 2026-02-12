# GPU Experiments for TropicalNN.jl

This directory contains scripts for testing and benchmarking GPU acceleration via cuPDLP.jl.

## Prerequisites

- NVIDIA GPU with CUDA support (compute capability ≥ 3.5)
- CUDA toolkit installed (version 11.0+)
- NVIDIA drivers
- Julia 1.10+
- SSH access to NVIDIA GPU machine

## Setup Instructions

### 1. Clone/Copy the Repository

```bash
# On the NVIDIA machine
git clone <repository-url>
cd gpu_tropical

# Or if copying via scp:
scp -r gpu_tropical/ user@nvidia-machine:~/
ssh user@nvidia-machine
cd gpu_tropical
```

### 2. Install TropicalNN.jl

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs all base dependencies (Oscar, JuMP, etc.)

### 3. Install GPU Dependencies

```bash
julia --project=. scripts/setup_gpu.jl
```

This script will:
- Install CUDA.jl
- Install cuPDLP.jl from GitHub
- Verify CUDA is functional
- Report GPU info (device name, memory, CUDA version)

**Expected output:**
```
✓ CUDA is functional!
  CUDA version: 12.0
  GPU device: NVIDIA A100-SXM4-40GB
  GPU memory: 40.0 GB
```

### 4. Test GPU Functionality

```bash
julia --project=. scripts/test_gpu.jl
```

This verifies:
- CUDA is available
- GPU path runs without errors
- Results match CPU implementation

### 5. Run Benchmarks

```bash
julia --project=. scripts/benchmark_gpu.jl
```

This benchmarks CPU vs GPU performance across different polynomial sizes.

**Expected speedups:** 3-10x for polynomials with 50+ monomials.

## Usage Examples

### Example 1: Simple GPU Test

```julia
using TropicalNN
using Oscar

# Check GPU is available
println("CUDA available: ", cuda_available())

# Create polynomial
R = Oscar.tropical_semiring(typeof(max))
f = TropicalPuiseuxPoly([R(1), R(2)], [[1//1, 0//1], [0//1, 1//1]], false)

# Run with GPU
regions = enum_linear_regions(f; use_gpu=true)
```

### Example 2: Convert MLP and Use GPU

```julia
using TropicalNN

# Create MLP
dims = [2, 4, 1]  # 2 inputs, 1 hidden layer (4 neurons), 1 output
W, b, t = random_mlp(dims)

# Convert to tropical (this step is CPU-only currently)
trop_func = mlp_to_trop(W, b, t)

# Enumerate regions with GPU (if using tropical polynomial)
if isa(trop_func, TropicalPuiseuxPoly)
    regions = enum_linear_regions(trop_func; use_gpu=true)
end
```

## Troubleshooting

### cuPDLP.jl Not in Registry

cuPDLP.jl is not yet in the Julia General registry. Install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/jinwen-yang/cuPDLP.jl")
```

### CUDA Not Functional

Check:
1. **GPU visible:** `nvidia-smi` should show your GPU
2. **CUDA installed:** Check `/usr/local/cuda` exists
3. **Drivers:** `nvidia-smi` shows driver version
4. **Julia can see CUDA:**
   ```julia
   using CUDA
   CUDA.versioninfo()
   ```

### Extension Not Loading

If the TropicalNNCUDAExt extension doesn't load:

```julia
# Check if extension is loaded
using TropicalNN
println(Base.get_extension(TropicalNN, :TropicalNNCUDAExt))
```

Should show the extension module. If `nothing`, CUDA.jl or cuPDLP.jl not installed.

### cuPDLP Solver Errors

cuPDLP.jl is experimental. If you get solver errors:
1. Check cuPDLP GitHub for issues/updates
2. Try simpler problems first (fewer monomials)
3. Use CPU fallback: `use_gpu=false`

## Expected Performance

Based on plan estimates:

| Problem Size | CPU Time | GPU Time | Speedup |
|-------------|----------|----------|---------|
| 20 monomials, 2D | ~500ms | ~150ms | 3x |
| 50 monomials, 2D | ~5s | ~800ms | 6x |
| 100 monomials, 2D | ~20s | ~2.5s | 8x |

**Note:** Actual speedups depend on GPU model, problem structure, and numerical conditioning.

## Files

- `setup_gpu.jl` - Install CUDA.jl and cuPDLP.jl
- `test_gpu.jl` - Verify GPU works correctly
- `benchmark_gpu.jl` - Compare CPU vs GPU performance
- `README_GPU.md` - This file

## Current Limitations

1. **cuPDLP.jl status:** Experimental, may have API changes
2. **Mac support:** None yet (CUDA only, Metal.jl support planned)
3. **Batch size:** Currently processes LPs sequentially (batching planned)
4. **Functions:** Only `enum_linear_regions()` has GPU support
   - `enum_linear_regions_rat()` - planned
   - `monomial_strong_elim()` - planned

## Getting Help

- TropicalNN.jl issues: https://github.com/yourusername/gpu_tropical/issues
- cuPDLP.jl issues: https://github.com/jinwen-yang/cuPDLP.jl/issues
- CUDA.jl docs: https://cuda.juliagpu.org/

## Citation

If you use GPU acceleration in your research:

```bibtex
@article{cupdlp2024,
  title={cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid Gradient for Linear Programming in Julia},
  author={Lu, Haihao and Yang, Jinwen},
  journal={Operations Research},
  year={2025},
  url={https://github.com/jinwen-yang/cuPDLP.jl}
}
```
