############### Utilities ###############

@doc raw"""
    linearmap_matrices(f::Signomial)

Returns the matrix of coefficients of the linear maps operating on the polyhedra of a tropical polynomial.
"""
function linearmap_matrices(f::Signomial)
    f = dedup_monomials(f)
    if length(f) == 0
        return zeros(Float64, 0, nvars(f)), Any[]
    end

    linear_maps_acc = Vector{Vector{Any}}()
    exponents_acc = Vector{Vector{Float64}}()
    coefficients_acc = Vector{Any}()
    for i in Base.eachindex(f)
        exp_i = get_exp(f, i)
        coeff_i = get_coeff(f, i)
        # we only want the linear maps that are realised
        if Oscar.is_fulldimensional(polyhedron(f, i, OscarMode()))
            linear_map = [Rational(coeff_i), collect(exp_i)]
            # we are only interested in the unique linear map
            if !(linear_map in linear_maps_acc)
                push!(exponents_acc, linear_map[2])
                push!(coefficients_acc, linear_map[1])
                push!(linear_maps_acc, linear_map)
            end
        end
    end
    if isempty(exponents_acc)
        return zeros(Float64, 0, nvars(f)), Any[]
    end
    A = mapreduce(permutedims, vcat, [Float64.(row) for row in exponents_acc])
    b = vec(coefficients_acc)
    return A, b
end

@doc raw"""
    linearmap_matrices(f::RationalSignomial)

Returns the matrix of coefficients of the linear maps operating on the polyhedra of a tropical rational map.
"""
function linearmap_matrices(f::RationalSignomial)
    Anum, bnum = linearmap_matrices(f.num)
    Aden, bden = linearmap_matrices(f.den)
    return (Anum, Aden), (bnum, bden)
end

@doc raw"""
    tilde_matrices(A::Matrix)

Finds all of the transformed 'tilde' matrices whose Hoffman constants are considered when obtaining the Hoffman constant of the corresponding tropical polynomial.
"""
function tilde_matrices(A::Matrix)
    m, n = size(A)
    ones_vector = ones(m, 1)
    return [A - ones_vector * reshape(A[row, :], (1, n)) for row in 1:m]
end

@doc raw"""
    tilde_matrices(As::Tuple{Matrix, Matrix})

Finds all of the transformed 'tilde' matrices whose Hoffman constants are considered when obtaining the Hoffman constant of the corresponding tropical rational map.
"""
function tilde_matrices(As::Tuple{Matrix, Matrix})
    m_1, n_1 = size(As[1])
    n_2 = size(As[2], 2)
    n_1 == n_2 ||
        throw(DimensionMismatch("Numerator and denominator matrices must have the same number of columns, got $n_1 and $n_2"))
    m_2 = size(As[2])[1]
    return [vcat(As[1] .- As[1][row_num:row_num, :],
                As[2] .- As[2][row_den:row_den, :])
            for row_den in 1:m_2, row_num in 1:m_1]
end

@doc raw"""
    tilde_vectors(b::Vector)

Find the transformed vectors used to determine the effective radius of a tropical polynomial
"""
function tilde_vectors(b::Vector)
    return [b - b[row] * ones(length(b)) for row in 1:length(b)]
end

@doc raw"""
    positive_component(b::Vector)

Returns the vector with its negative entries set to zero.
"""
function positive_component(b::Vector)
    return vec([max(0, entry) for entry in b])
end

function _empty_hoff_return(return_matrices::Bool, A, b)
    if return_matrices
        return Inf, A, b
    else
        return Inf
    end
end

function _t_matrices_or_inf(A, b, return_matrices::Bool)
    t_matrices = tilde_matrices(A)
    isempty(t_matrices) && return nothing, _empty_hoff_return(return_matrices, A, b)
    return t_matrices, nothing
end

############### Hoffman Algorithms ###############

@doc raw"""
    surjectivity_test(A::Matrix) -> (x_val, t_val)

Test whether the matrix `A` is *A-surjective* — a condition arising in Hoffman constant computation
for tropical polynomials.

A matrix ``\tilde{A}`` (a "tilde matrix" derived from the linear-map matrix ``A`` via
[`tilde_matrices`](@ref)) is A-surjective if the image of the positive orthant under ``\tilde{A}^T``
intersects every open halfspace through the origin; equivalently, there is no direction in which
all rows of ``\tilde{A}`` have non-positive dot product.  A-surjectivity is the precondition under
which the Hoffman constant ``H(\tilde{A})`` is finite (see [`exact_hoff`](@ref)).

The test solves the LP

```
min  t
s.t. ‖Aᵀx‖₁ ≤ t,  sum(x) = 1,  x ≥ 0
```

# Arguments
- `A::Matrix`: A tilde matrix (typically one element of the vector returned by [`tilde_matrices`](@ref)).

# Returns
- `(x_val, t_val)`: the optimal primal variable `x` (a probability vector over the rows of `A`) and
  the optimal objective value `t`.  If `t_val > 0` the matrix is A-surjective and
  ``H(\tilde{A}) = 1 / t_{\min}`` where ``t_{\min}`` is the minimum over all feasible probability
  vectors; if `t_val == 0` then `A` is **not** A-surjective and the Hoffman constant is infinite.
"""
function _surjectivity_scale(A::Matrix)
    return norm(A, Inf)
end

function _surjectivity_objective_tol(A::Matrix, tol::Float64)
    iszero(tol) && return 0.0
    return tol * _surjectivity_scale(A)
end

function surjectivity_test(A::Matrix; tol::Float64 = 1e-10)
    n = size(A, 2)
    m = size(A, 1)
    tol >= 0 || throw(ArgumentError("tol must be nonnegative, got $tol"))
    scale = _surjectivity_scale(A)
    A_lp = iszero(scale) ? A : A ./ scale

    # setting up the model
    model = Model(GLPK.Optimizer)
    set_silent(model)
    @variable(model, x[1:m] >= 0)
    @variable(model, t)
    @objective(model, Min, t)
    @constraint(model, [t; A_lp' * x] in MOI.NormOneCone(1 + n))
    @constraint(model, sum(x) == 1)

    # solving the model
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL ||
        throw(ErrorException("GLPK surjectivity LP ended with unexpected status $status"))

    x_val = value.(x)
    t_val = iszero(scale) ? value(t) : value(t) * scale

    # accounting for numerical errors without destroying scale covariance
    x_val = map(v -> abs(v) < tol ? 0.0 : v, x_val)
    t_val = abs(t_val) < _surjectivity_objective_tol(A, tol) ? 0.0 : t_val

    return x_val, t_val
end


@doc raw"""
    exact_hoff(A::Matrix)

Computes the exact Hoffman constant of the matrix `A` by iterating over all subsets of rows
and solving an A-surjectivity LP for each subset.

!!! warning "Exponential complexity"
    This function iterates over all ``2^m`` subsets of the ``m`` rows of `A`.
    Use [`upper_hoff`](@ref) or [`lower_hoff`](@ref) for large matrices.
"""
function exact_hoff(A::Matrix; tol::Float64 = 1e-10)
    m = size(A, 1)
    H = -Inf
    found_surjective = false
    # iterating over sub-matrices of A
    for j in 1:m
        for subset in Combinatorics.combinations(1:m, j)
            AA = A[subset, :]
            # solving the optimisation problem
            y, t = surjectivity_test(AA; tol = tol)
            if t > 0
                # in this case the subset is A-surjective
                H = max(H, 1 / t)
                found_surjective = true
            end
        end
    end
    if found_surjective
        return H
    else
        # if no sub-matrix is A-surjective then the Hoffman constant is infinite
        return Inf
    end
end

@doc raw"""
    pvz_hoff(A::Matrix; return_certificates::Bool=false)

Computes the exact Hoffman constant of the matrix `A` using the
Pena--Vera--Zuluaga pruning algorithm.

Instead of testing every row subset independently, PVZ keeps a frontier of
candidate subsets.  When a candidate is `A`-surjective, all of its subsets are
known to be covered by that certificate and can be removed from the frontier.
When a candidate is not `A`-surjective, the LP returns a nonzero support that
must be broken; every remaining candidate containing that support is replaced
by smaller candidates obtained by deleting one support index.

The algorithm maintains:

- `F`: row subsets certified to be `A`-surjective;
- `I`: row subsets certified not to be `A`-surjective;
- `J`: candidate row subsets not yet certified.

For each candidate subset, it solves the same A-surjectivity LP as
[`exact_hoff`](@ref).  Surjective candidates certify all of their subsets, and
non-surjective candidates are split using the support of the LP certificate.
This is still an exact algorithm, but it can avoid many LP solves compared with
brute-force subset enumeration.
"""
function pvz_hoff(A::Matrix; return_certificates::Bool = false, tol::Float64 = 1e-10)
    m = size(A, 1)
    H = -Inf
    found_surjective = false

    # Start from the full row set and let the PVZ updates shrink the frontier.
    F = Vector{Vector{Int}}()
    I = Vector{Vector{Int}}()
    candidates = Vector{Vector{Int}}()
    if m > 0
        push!(candidates, collect(1:m))
    end

    while !isempty(candidates)
        J = pop!(candidates)
        # Test one frontier candidate with the PVZ A-surjectivity LP.
        x, t = surjectivity_test(A[J, :]; tol = tol)

        if t > 0
            # A surjective set certifies itself and every subset below it.
            push!(F, J)
            H = max(H, 1 / t)
            found_surjective = true

            filter!(candidate -> !issubset(candidate, J), candidates)
        else
            # The positive support is an obstruction that no candidate may keep intact.
            support = [J[index] for index in eachindex(J) if x[index] > tol]
            support = sort(unique(support))
            push!(I, support)

            # Pull out all current candidates containing the obstruction support.
            containing_support = Vector{Vector{Int}}()
            push!(containing_support, J)
            remaining_candidates = Vector{Vector{Int}}()
            for candidate in candidates
                if issubset(support, candidate)
                    push!(containing_support, candidate)
                else
                    push!(remaining_candidates, candidate)
                end
            end
            candidates = remaining_candidates

            # Replace each obstructed candidate by the children that delete one support index.
            for candidate in containing_support
                for index in support
                    reduced_candidate = setdiff(candidate, index)
                    isempty(reduced_candidate) && continue
                    any(F_set -> issubset(reduced_candidate, F_set), F) && continue
                    reduced_candidate in candidates || push!(candidates, reduced_candidate)
                end
            end
        end
    end

    hoff_const = found_surjective ? H : Inf
    if return_certificates
        return hoff_const, F, I
    else
        return hoff_const
    end
end

@doc raw"""
    upper_hoff(A::Matrix)

Computes an upper bound on Hoffman constant of the matrix `A` by using the lowest singular value as a proxy for the optimal value of the optimisation problem for A-surjectivity.
"""
function upper_hoff(A::Matrix)
    m, n = size(A)
    HU = -Inf
    found_surjective = false
    # iterating over sub-matrices of A
    for j in 1:m
        for subset in Combinatorics.combinations(1:m, j)
            AJ = A[subset, :]
            # only considering full rank sub-matrices
            if rank(AJ) == min(j, n)
                # compute lowest singular value of the sub-matrix
                p_J = minimum(svdvals(AJ))
                if p_J > 0
                    HU = max(HU, sqrt(length(subset)) / p_J)
                    found_surjective = true
                end
            end
        end
    end
    if found_surjective
        return HU
    else
        # if no sub-matrix is A-surjective then the Hoffman constant is infinite
        return Inf
    end
end

@doc raw"""
    lower_hoff(A::Matrix,num_samples::Int=10)

Computes a lower bound on Hoffman constant of the matrix `A` by only considering a fixed number of random sub-matrices of A.
"""
function lower_hoff(A::Matrix, num_samples::Int = 10; tol::Float64 = 1e-10)
    m, n = size(A)
    HL = 0.0
    # if the number of sub-matrices we are considering exceeds the total number of sub-matrices in A
    # we can just use the exact method with no additional computational resources
    if num_samples >= 2^m
        return exact_hoff(A; tol = tol)
    else
        for i in 1:num_samples
            # consider random sub-matrices
            K = rand(1:m)
            J = sort(unique(rand(1:m, K)))
            AJ = A[J, :]
            x, t = surjectivity_test(AJ; tol = tol)
            if t > 0
                HL = max(HL, 1 / t)
            end
        end
    end
    return HL
end

@doc raw"""
    exact_hoff(f::Union{Signomial,RationalSignomial};return_matrices::Bool=false)

Returns the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function exact_hoff(f::Union{Signomial, RationalSignomial};
        return_matrices::Bool = false, tol::Float64 = 1e-10)
    hoff_const = 0
    A, b = linearmap_matrices(f)
    t_matrices, empty_return = _t_matrices_or_inf(A, b, return_matrices)
    empty_return !== nothing && return empty_return
    for tilde_matrix in t_matrices
        # constant is taken to be the maximum over each of the tilde matrices
        hoff_const = max(hoff_const, exact_hoff(tilde_matrix; tol = tol))
    end
    if return_matrices
        return hoff_const, A, b
    else
        return hoff_const
    end
end

@doc raw"""
    pvz_hoff(f::Union{Signomial,RationalSignomial};return_matrices::Bool=false)

Returns the exact value of the Hoffman constant of a given tropical polynomial
or tropical rational map, using [`pvz_hoff`](@ref) on each transformed tilde
matrix.
"""
function pvz_hoff(f::Union{Signomial, RationalSignomial};
        return_matrices::Bool = false, tol::Float64 = 1e-10)
    hoff_const = 0
    A, b = linearmap_matrices(f)
    t_matrices, empty_return = _t_matrices_or_inf(A, b, return_matrices)
    empty_return !== nothing && return empty_return
    for tilde_matrix in t_matrices
        hoff_const = max(hoff_const, pvz_hoff(tilde_matrix; tol = tol))
    end
    if return_matrices
        return hoff_const, A, b
    else
        return hoff_const
    end
end

@doc raw"""
    pvz_hoff(f::Union{Signomial,RationalSignomial};return_matrices::Bool=false)

Returns the exact value of the Hoffman constant of a given tropical polynomial
or tropical rational map, using [`pvz_hoff`](@ref) on each transformed tilde
matrix.
"""
function pvz_hoff(f::Union{Signomial, RationalSignomial};
        return_matrices::Bool = false, tol::Float64 = 1e-10)
    hoff_const = 0
    A, b = linearmap_matrices(f)
    for tilde_matrix in tilde_matrices(A)
        hoff_const = max(hoff_const, pvz_hoff(tilde_matrix; tol = tol))
    end
    if return_matrices
        return hoff_const, A, b
    else
        return hoff_const
    end
end

@doc raw"""
    upper_hoff(f::Union{Signomial,RationalSignomial};return_matrices::Bool=false)

Returns an upper bound on the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function upper_hoff(f::Union{Signomial, RationalSignomial}; return_matrices::Bool = false)
    hoff_upper = 0
    A, b = linearmap_matrices(f)
    t_matrices, empty_return = _t_matrices_or_inf(A, b, return_matrices)
    empty_return !== nothing && return empty_return
    for tilde_matrix in t_matrices
        # to ensure we have an upper bound we need to take the maximum across all upper bounds
        hoff_upper = max(hoff_upper, upper_hoff(tilde_matrix))
    end
    if return_matrices
        return hoff_upper, A, b
    else
        return hoff_upper
    end
end

@doc raw"""
    lower_hoff(f::Union{Signomial,RationalSignomial},num_samples::Int=10)

Returns a lower bound on the exact value of the Hoffman constant of a given tropical polynomial or tropical rational map.
"""
function lower_hoff(f::Union{Signomial, RationalSignomial},
        num_samples::Int = 10; return_matrices::Bool = false, tol::Float64 = 1e-10)
    A, b = linearmap_matrices(f)
    t_matrices, empty_return = _t_matrices_or_inf(A, b, return_matrices)
    empty_return !== nothing && return empty_return
    # if we are taking more samples than there are submatrices we are using exact
    # computations so we can take a maximum over the Hoffman constants
    # The Hoffman constant of the tropical function is max_k H(tilde_matrix_k).
    # Each lower_hoff(tilde_matrix_k, ...) is a lower bound on H(tilde_matrix_k).
    # max over lower bounds is still a lower bound on the overall max, regardless of
    # whether we are in the exact or sampling regime.
    hoff_lower = 0.0
    for tilde_matrix in t_matrices
        hoff_lower = max(hoff_lower, lower_hoff(tilde_matrix, num_samples; tol = tol))
    end
    if return_matrices
        return hoff_lower, A, b
    else
        return hoff_lower
    end
end

############### Effective Radius ###############

@doc raw"""
    exact_er(f::Signomial)

Provides an upper bound on the effective radius of a tropical polynomial using exact Hoffman constant computations.
"""
function exact_er(f::Signomial)
    hoff_const, A, b = exact_hoff(f, return_matrices = true)
    isinf(hoff_const) && return Inf
    tilde_bs = tilde_vectors(b)
    return hoff_const *
           maximum([norm(positive_component(tilde_b), Inf) for tilde_b in tilde_bs])
end

@doc raw"""
    upper_er(f::Signomial)

Provides an upper bound on the effective radius of a tropical polynomial using upper bound approximations of the Hoffman constant.
"""
function upper_er(f::Signomial)
    hoff_upper, A, b = upper_hoff(f, return_matrices = true)
    isinf(hoff_upper) && return Inf
    tilde_bs = tilde_vectors(b)
    return hoff_upper *
           maximum([norm(positive_component(tilde_b), Inf) for tilde_b in tilde_bs])
end

@doc raw"""
    exact_er(f::RationalSignomial)

Provides an upper bound on the effective radius of a tropical rational map using exact Hoffman constant computations.
"""
function exact_er(f::RationalSignomial)
    hoff_const, A, b = exact_hoff(f, return_matrices = true)
    isinf(hoff_const) && return Inf
    return hoff_const * max(maximum(b[1]) - minimum(b[1]), maximum(b[2]) - minimum(b[2]))
end

@doc raw"""
    upper_er(f::RationalSignomial)

Provides an upper bound on the effective radius of a tropical rational map using upper bound approximations of the Hoffman constant.
"""
function upper_er(f::RationalSignomial)
    hoff_upper, A, b = upper_hoff(f, return_matrices = true)
    isinf(hoff_upper) && return Inf
    return hoff_upper * max(maximum(b[1]) - minimum(b[1]), maximum(b[2]) - minimum(b[2]))
end
