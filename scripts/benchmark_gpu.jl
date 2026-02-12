#!/usr/bin/env julia
#
# GPU Benchmark Script for TropicalNN.jl
#
# Benchmarks CPU vs GPU performance for enum_linear_regions
# across different polynomial sizes.

using TropicalNN
using Oscar
using Printf

println("=" ^ 70)
println("TropicalNN.jl GPU Benchmark")
println("=" ^ 70)
println()

# Check GPU availability
if !cuda_available()
    println("✗ CUDA not available. Running CPU-only benchmark.")
    gpu_enabled = false
else
    println("✓ CUDA available: GPU acceleration enabled")
    println("  GPU: ", CUDA.name(CUDA.device()))
    gpu_enabled = true
end
println()

# Benchmark configuration
R = Oscar.tropical_semiring(typeof(max))
test_sizes = [5, 10, 20, 50, 100]  # Number of monomials

println("Benchmark configuration:")
println("  Polynomial sizes: $test_sizes monomials")
println("  Dimensions: 2D")
println()

# Results storage
results = []

for n in test_sizes
    println("Testing n=$n monomials...")

    # Generate random tropical polynomial
    exps = [[Rational(i), Rational(j)] for i in 0:n-1 for j in 0:0]
    exps = exps[1:min(n, length(exps))]
    coeffs = [R(rand()) for _ in 1:length(exps)]
    f = TropicalPuiseuxPoly(coeffs, exps, false)

    # Warmup
    enum_linear_regions(f; use_gpu=false)

    # CPU benchmark
    t_cpu = @elapsed begin
        regions_cpu = enum_linear_regions(f; use_gpu=false)
    end
    n_regions_cpu = length(regions_cpu)

    # GPU benchmark (if available)
    if gpu_enabled
        try
            # Warmup
            enum_linear_regions(f; use_gpu=true)

            t_gpu = @elapsed begin
                regions_gpu = enum_linear_regions(f; use_gpu=true)
            end
            n_regions_gpu = length(regions_gpu)

            speedup = t_cpu / t_gpu

            push!(results, (n, t_cpu, t_gpu, speedup, n_regions_cpu, n_regions_gpu))

            @printf("  CPU: %.3f ms | GPU: %.3f ms | Speedup: %.2fx\n",
                    t_cpu*1000, t_gpu*1000, speedup)
        catch e
            println("  CPU: $(round(t_cpu*1000, digits=3)) ms | GPU: FAILED")
            println("    Error: $e")
            push!(results, (n, t_cpu, NaN, NaN, n_regions_cpu, 0))
        end
    else
        println("  CPU: $(round(t_cpu*1000, digits=3)) ms")
        push!(results, (n, t_cpu, NaN, NaN, n_regions_cpu, 0))
    end
    println()
end

# Summary
println("=" ^ 70)
println("Benchmark Summary")
println("=" ^ 70)
println()
@printf("%-10s | %-12s | %-12s | %-10s | %-10s\n",
        "Monomials", "CPU (ms)", "GPU (ms)", "Speedup", "Regions")
println("-" ^ 70)

for (n, t_cpu, t_gpu, speedup, n_regions, _) in results
    if isnan(t_gpu)
        @printf("%-10d | %12.3f | %12s | %10s | %10d\n",
                n, t_cpu*1000, "N/A", "N/A", n_regions)
    else
        @printf("%-10d | %12.3f | %12.3f | %9.2fx | %10d\n",
                n, t_cpu*1000, t_gpu*1000, speedup, n_regions)
    end
end

if gpu_enabled && any(!isnan(r[3]) for r in results)
    println()
    println("GPU achieves best speedup at n=$(results[end][1]): $(round(results[end][4], digits=2))x")
end

println()
println("=" ^ 70)
println("Benchmark complete!")
println("=" ^ 70)
