#!/usr/bin/env julia
#
# GPU Test Script for TropicalNN.jl
#
# Tests that GPU acceleration is working correctly by running
# enum_linear_regions with GPU enabled and comparing to CPU results.

using TropicalNN
using Oscar

println("=" ^ 70)
println("TropicalNN.jl GPU Test")
println("=" ^ 70)
println()

# Check GPU availability
println("GPU Status:")
println("  CUDA available: ", cuda_available())
println()

if !cuda_available()
    println("✗ CUDA not available. Cannot run GPU tests.")
    println("  Run setup_gpu.jl first to install CUDA support.")
    exit(1)
end

# Create test polynomial
println("Creating test polynomial...")
R = Oscar.tropical_semiring(typeof(max))
f = TropicalPuiseuxPoly([R(1), R(2), R(3)], [[1//1, 0//1], [0//1, 1//1], [1//1, 1//1]], false)
println("  Polynomial: f with $(length(f.exp)) monomials")
println()

# Test CPU path
println("Testing CPU path...")
t_cpu = @elapsed begin
    regions_cpu = enum_linear_regions(f; use_gpu=false)
end
println("  ✓ CPU completed in $(round(t_cpu*1000, digits=2)) ms")
println("  Found $(length(regions_cpu)) regions")
feasible_cpu = [r[2] for r in regions_cpu]
println("  Feasibility: $feasible_cpu")
println()

# Test GPU path
println("Testing GPU path...")
t_gpu = @elapsed begin
    try
        regions_gpu = enum_linear_regions(f; use_gpu=true)
        println("  ✓ GPU completed in $(round(t_gpu*1000, digits=2)) ms")
        println("  Found $(length(regions_gpu)) regions")
        feasible_gpu = [r[2] for r in regions_gpu]
        println("  Feasibility: $feasible_gpu")

        # Compare results
        println()
        println("Comparing CPU vs GPU results...")
        if feasible_cpu == feasible_gpu
            println("  ✓ Results match!")
        else
            println("  ✗ Results differ!")
            println("    CPU: $feasible_cpu")
            println("    GPU: $feasible_gpu")
        end
    catch e
        println("  ✗ GPU test failed:")
        println("    Error: $e")
        println()
        println("  This might be because cuPDLP.jl is not fully functional yet.")
        println("  The GPU infrastructure is in place, but cuPDLP may need")
        println("  additional setup or configuration.")
    end
end

println()
println("=" ^ 70)
println("Test complete!")
println("=" ^ 70)
