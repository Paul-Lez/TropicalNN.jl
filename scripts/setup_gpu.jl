#!/usr/bin/env julia
#
# GPU Setup Script for TropicalNN.jl
#
# This script installs CUDA.jl and cuPDLP.jl to enable GPU acceleration.
# Run this on an NVIDIA GPU machine before running experiments.

using Pkg

println("=" ^ 70)
println("TropicalNN.jl GPU Setup")
println("=" ^ 70)
println()

# Add CUDA.jl
println("Installing CUDA.jl...")
try
    Pkg.add("CUDA")
    println("✓ CUDA.jl installed")
catch e
    println("✗ Failed to install CUDA.jl: $e")
    exit(1)
end

# Add cuPDLP.jl (from GitHub since it's not in General registry yet)
println("\nInstalling cuPDLP.jl from GitHub...")
try
    Pkg.add(url="https://github.com/jinwen-yang/cuPDLP.jl")
    println("✓ cuPDLP.jl installed")
catch e
    println("✗ Failed to install cuPDLP.jl: $e")
    println("  (This is expected if the package structure has changed)")
    println("  Try: Pkg.add(name=\"cuPDLP\", url=\"https://github.com/jinwen-yang/cuPDLP.jl\")")
end

# Verify CUDA is functional
println("\nVerifying CUDA installation...")
using CUDA

if CUDA.functional()
    println("✓ CUDA is functional!")
    println("  CUDA runtime version: ", CUDA.runtime_version())
    println("  GPU device: ", CUDA.name(CUDA.device()))
    println("  GPU memory: ", round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=2), " GB")
else
    println("✗ CUDA is not functional")
    println("  Make sure you have:")
    println("  1. NVIDIA GPU with compute capability ≥ 3.5")
    println("  2. CUDA toolkit installed (version 11.0+)")
    println("  3. NVIDIA drivers installed")
    exit(1)
end

# Precompile everything
println("\nPrecompiling packages...")
Pkg.precompile()

println("\n" * "=" ^ 70)
println("Setup complete! You can now run GPU experiments.")
println("=" ^ 70)
