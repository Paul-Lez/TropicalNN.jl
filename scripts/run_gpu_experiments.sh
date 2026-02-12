#!/bin/bash
#
# Quick-start script for GPU experiments
# Run this on an NVIDIA GPU machine
#

set -e

echo "=========================================="
echo "TropicalNN.jl GPU Experiments Quick Start"
echo "=========================================="
echo ""

# Check Julia is installed
if ! command -v julia &> /dev/null; then
    echo "✗ Julia not found. Please install Julia 1.10+"
    exit 1
fi

echo "✓ Julia found: $(julia --version)"
echo ""

# Setup project
echo "Step 1: Installing TropicalNN.jl dependencies..."
julia --project=. -e 'using Pkg; Pkg.instantiate()' || exit 1
echo ""

# Setup GPU
echo "Step 2: Installing GPU dependencies (CUDA.jl + cuPDLP.jl)..."
julia --project=. scripts/setup_gpu.jl || exit 1
echo ""

# Test GPU
echo "Step 3: Testing GPU functionality..."
julia --project=. scripts/test_gpu.jl || exit 1
echo ""

# Benchmark
echo "Step 4: Running benchmarks..."
julia --project=. scripts/benchmark_gpu.jl || exit 1
echo ""

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
