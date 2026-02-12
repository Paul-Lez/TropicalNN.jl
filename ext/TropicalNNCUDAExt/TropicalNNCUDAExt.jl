module TropicalNNCUDAExt

using TropicalNN
using CUDA
using cuPDLP
using JuMP
import MathOptInterface as MOI

# Override cuda_available() to check if CUDA is functional
function TropicalNN.cuda_available()::Bool
    return CUDA.functional()
end

"""
    TropicalNN.batch_feasibility_cupdlp(A_list, b_list)

GPU implementation of batch polyhedral feasibility checking using cuPDLP.jl.

For each polyhedron defined by Ax ≤ b, solves the LP feasibility problem:
    minimize    0
    subject to  Ax ≤ b

Returns a vector of booleans indicating whether each polyhedron is feasible.

# Implementation Notes
- Uses JuMP with cuPDLP.Optimizer backend
- Solves LP on GPU using cuPDLP's PDHG algorithm
- Timeout set to 10 seconds per LP solve

# Arguments
- `A_list::Vector{Matrix{Float64}}`: Constraint matrices
- `b_list::Vector{Vector{Float64}}`: RHS vectors

# Returns
- `Vector{Bool}`: Feasibility status for each polyhedron
"""
function TropicalNN.batch_feasibility_cupdlp(
    A_list::Vector{Matrix{Float64}},
    b_list::Vector{Vector{Float64}}
)
    if !CUDA.functional()
        error("CUDA is not functional. Cannot use GPU acceleration.")
    end

    n_poly = length(A_list)
    results = Vector{Bool}(undef, n_poly)

    for i in 1:n_poly
        A = A_list[i]
        b = b_list[i]

        # Check dimensions
        m, n = size(A)  # m constraints, n variables
        @assert length(b) == m "Dimension mismatch: A has $m rows but b has $(length(b)) elements"

        try
            # Set up LP problem using JuMP
            model = Model(cuPDLP.Optimizer)

            # Silence solver output
            set_silent(model)

            # Set time limit (10 seconds)
            set_time_limit_sec(model, 10.0)

            # Variables
            @variable(model, x[1:n])

            # Objective: minimize 0 (feasibility problem)
            @objective(model, Min, 0.0)

            # Constraints: Ax ≤ b
            @constraint(model, A * x .<= b)

            # Solve on GPU
            optimize!(model)

            # Check if we found a feasible solution
            status = termination_status(model)
            results[i] = (status == MOI.OPTIMAL ||
                         status == MOI.FEASIBLE_POINT ||
                         status == MOI.ALMOST_OPTIMAL ||
                         status == MOI.ALMOST_FEASIBLE)

        catch e
            # If solve fails, assume infeasible
            @warn "GPU solve failed for polyhedron $i: $e" maxlog=5
            results[i] = false
        end
    end

    return results
end

end # module
