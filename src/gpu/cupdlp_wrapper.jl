# GPU LP solving via cuPDLP.jl for NVIDIA CUDA GPUs
#
# This module provides wrapper functions for cuPDLP.jl to solve
# linear programming feasibility problems on GPU, which is used
# to check if polyhedra Ax ≤ b are non-empty.

"""
    cuda_available()::Bool

Check if CUDA GPU is available and functional.

Returns `true` if CUDA.jl extension is loaded and a functional CUDA device is detected,
`false` otherwise. This is a stub that returns `false` - the actual implementation
is provided by the TropicalNNCUDAExt extension when CUDA is available.

# Example
```julia
if cuda_available()
    println("GPU acceleration enabled")
else
    println("Falling back to CPU")
end
```
"""
function cuda_available()::Bool
    return false  # Overridden by extension when CUDA is loaded
end

"""
    batch_feasibility_cupdlp(A_list::Vector{Matrix{Float64}}, b_list::Vector{Vector{Float64}})

Check feasibility of multiple polyhedra Ax ≤ b on GPU using cuPDLP.jl.

This is a stub implementation that throws an error. The actual GPU implementation
is provided by the TropicalNNCUDAExt extension when CUDA and cuPDLP are available.

# Arguments
- `A_list`: Vector of constraint matrices
- `b_list`: Vector of RHS vectors

# Returns
- `Vector{Bool}`: Feasibility status for each polyhedron

# Example
```julia
A_list = [rand(10, 5) for _ in 1:100]
b_list = [rand(10) for _ in 1:100]
results = batch_feasibility_cupdlp(A_list, b_list)  # GPU accelerated
```
"""
function batch_feasibility_cupdlp(A_list::Vector{Matrix{Float64}}, b_list::Vector{Vector{Float64}})
    error("cuPDLP GPU support not available. Please install CUDA.jl and cuPDLP.jl and ensure a CUDA GPU is available.")
end

