# Matrix construction.

@doc raw"""
    linearmap_matrices(f::Signomial; mode::LinearRegionsCalculationMode=OscarMode())

Return exponent matrix `A` and coefficient vector `b` for the non-redundant monomials of `f`.
"""
function linearmap_matrices(f::Signomial; mode::LinearRegionsCalculationMode = OscarMode())
    f = dedup_monomials(f)
    if length(f) == 0
        return zeros(Float64, 0, nvars(f)), Any[]
    end

    linear_maps_acc = Vector{Vector{Any}}()
    exponents_acc = Vector{Vector{Float64}}()
    coefficients_acc = Vector{Any}()
    competitors = collect(Base.eachindex(f))
    for i in Base.eachindex(f)
        exp_i = get_exp(f, i)
        coeff_i = get_coeff(f, i)
        poly = polyhedron(f, i, mode; competitors = competitors)
        if is_full_dimensional(poly; mode = mode)
            linear_map = [Rational(coeff_i), collect(exp_i)]
            if !(linear_map in linear_maps_acc)
                push!(exponents_acc, linear_map[2])
                push!(coefficients_acc, linear_map[1])
                push!(linear_maps_acc, linear_map)
            end
        else
            deleteat!(competitors, searchsortedfirst(competitors, i))
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
    linearmap_matrices(f::RationalSignomial; mode::LinearRegionsCalculationMode=OscarMode())

Return the exponent matrices and coefficient vectors for the numerator and
denominator of `f`.
"""
function linearmap_matrices(
        f::RationalSignomial;
        mode::LinearRegionsCalculationMode = OscarMode()
)
    Anum, bnum = linearmap_matrices(f.num; mode = mode)
    Aden, bden = linearmap_matrices(f.den; mode = mode)
    return (Anum, Aden), (bnum, bden)
end

@doc raw"""
    tilde_matrices(A::Matrix)

Return `A - ones(m) * A[i, :]'` for each row `i` of `A`.
"""
function tilde_matrices(A::Matrix)
    m, n = size(A)
    ones_vector = ones(m, 1)
    return [A - ones_vector * reshape(A[row, :], (1, n)) for row in 1:m]
end

@doc raw"""
    tilde_matrices(As::Tuple{Matrix, Matrix})

Return the transformed matrices for all numerator-denominator row
pairs.
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

Return `b - b[i] * ones(length(b))` for each index `i`.
"""
function tilde_vectors(b::Vector)
    return [b - b[row] * ones(length(b)) for row in 1:length(b)]
end

@doc raw"""
    positive_component(b::Vector)

Return `max.(b, 0)` as a vector.
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

# Hoffman calculations.

function _surjectivity_scale(A::Matrix)
    return norm(A, Inf)
end

function _surjectivity_objective_tol(A::Matrix, tol::Float64)
    iszero(tol) && return 0.0
    return tol * _surjectivity_scale(A)
end

@doc raw"""
    surjectivity_test(A::Matrix; tol=1e-10) -> (v, t)

Test A-surjectivity with the GLPK problem
`min ‖A'v‖₁` subject to `sum(v) = 1` and `v ≥ 0`. Return `(v, t)`; the rows
are A-surjective when `t > 0`. `tol` removes small floating-point results.
"""
function surjectivity_test(A::Matrix; tol::Float64 = 1e-10)
    n = size(A, 2)
    m = size(A, 1)
    tol >= 0 || throw(ArgumentError("tol must be nonnegative, got $tol"))
    scale = _surjectivity_scale(A)
    A_lp = iszero(scale) ? A : A ./ scale

    model = Model(GLPK.Optimizer)
    set_silent(model)
    @variable(model, x[1:m] >= 0)
    @variable(model, t)
    @objective(model, Min, t)
    @constraint(model, [t; A_lp' * x] in MOI.NormOneCone(1 + n))
    @constraint(model, sum(x) == 1)

    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL ||
        throw(ErrorException("GLPK surjectivity LP ended with unexpected status $status"))

    x_val = value.(x)
    t_val = iszero(scale) ? value(t) : value(t) * scale

    # Apply a scale-covariant tolerance to the result.
    x_val = map(v -> abs(v) < tol ? 0.0 : v, x_val)
    t_val = abs(t_val) < _surjectivity_objective_tol(A, tol) ? 0.0 : t_val

    return x_val, t_val
end

@doc raw"""
    exact_hoff(A::Matrix; tol=1e-10)

Compute the infinity-norm Hoffman constant by testing every nonempty row
subset. The enumeration is exhaustive, and GLPK uses floating-point
arithmetic.
"""
function exact_hoff(A::Matrix; tol::Float64 = 1e-10)
    m = size(A, 1)
    H = -Inf
    found_surjective = false
    for j in 1:m
        for subset in Combinatorics.combinations(1:m, j)
            AA = A[subset, :]
            y, t = surjectivity_test(AA; tol = tol)
            if t > 0
                H = max(H, 1 / t)
                found_surjective = true
            end
        end
    end
    if found_surjective
        return H
    else
        return Inf
    end
end

@doc raw"""
    pvz_hoff(A::Matrix; return_certificates=false, tol=1e-10)

Compute the infinity-norm Hoffman constant with the Peña-Vera-Zuluaga algoritm.
With `return_certificates=true`, return `(H, F, I)` with the surjective sets
and obstruction supports. GLPK uses floating-point arithmetic.
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

Return the largest `sqrt(length(J)) / minimum(svdvals(A[J, :]))` over
nonempty full-rank row subsets `J`.
"""
function upper_hoff(A::Matrix)
    m, n = size(A)
    HU = -Inf
    found_surjective = false
    for j in 1:m
        for subset in Combinatorics.combinations(1:m, j)
            AJ = A[subset, :]
            if rank(AJ) == min(j, n)
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
        return Inf
    end
end

@doc raw"""
    lower_hoff(A::Matrix, num_samples::Int=10; tol=1e-10)

Return a lower bound from `num_samples` random nonempty row subsets. If
`num_samples >= 2^m`, call [`exact_hoff`](@ref).
"""
function lower_hoff(A::Matrix, num_samples::Int = 10; tol::Float64 = 1e-10)
    m, n = size(A)
    HL = 0.0
    # Enumerate all nonempty subsets when sampling would require more tests.
    if num_samples >= 2^m
        return exact_hoff(A; tol = tol)
    else
        for i in 1:num_samples
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
    exact_hoff(f::Union{Signomial,RationalSignomial};
               return_matrices=false, mode=OscarMode(), tol=1e-10)

Compute the Hoffman constant for the stored expression of `f` by exhaustive
subset enumeration. With `return_matrices=true`, return `(H, A, b)`. GLPK
uses floating-point arithmetic.
"""
function exact_hoff(f::Union{Signomial, RationalSignomial};
        return_matrices::Bool = false,
        mode::LinearRegionsCalculationMode = OscarMode(),
        tol::Float64 = 1e-10)
    hoff_const = 0
    A, b = linearmap_matrices(f; mode = mode)
    t_matrices, empty_return = _t_matrices_or_inf(A, b, return_matrices)
    empty_return !== nothing && return empty_return
    for tilde_matrix in t_matrices
        hoff_const = max(hoff_const, exact_hoff(tilde_matrix; tol = tol))
    end
    if return_matrices
        return hoff_const, A, b
    else
        return hoff_const
    end
end

@doc raw"""
    pvz_hoff(f::Union{Signomial,RationalSignomial};
             return_matrices=false, mode=OscarMode(), tol=1e-10)

Compute the Hoffman constant for the stored expression of `f` with the
Peña-Vera-Zuluaga algorithm. With `return_matrices=true`, return `(H, A, b)`.
"""
function pvz_hoff(f::Union{Signomial, RationalSignomial};
        return_matrices::Bool = false,
        mode::LinearRegionsCalculationMode = OscarMode(),
        tol::Float64 = 1e-10)
    hoff_const = 0
    A, b = linearmap_matrices(f; mode = mode)
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
    upper_hoff(f::Union{Signomial,RationalSignomial};
               return_matrices=false, mode=OscarMode())

Return a Hoffman-constant upper bound for the stored expression of `f`.
With `return_matrices=true`, return `(bound, A, b)`.
"""
function upper_hoff(
        f::Union{Signomial, RationalSignomial};
        return_matrices::Bool = false,
        mode::LinearRegionsCalculationMode = OscarMode()
)
    hoff_upper = 0
    A, b = linearmap_matrices(f; mode = mode)
    t_matrices, empty_return = _t_matrices_or_inf(A, b, return_matrices)
    empty_return !== nothing && return empty_return
    for tilde_matrix in t_matrices
        hoff_upper = max(hoff_upper, upper_hoff(tilde_matrix))
    end
    if return_matrices
        return hoff_upper, A, b
    else
        return hoff_upper
    end
end

@doc raw"""
    lower_hoff(f::Union{Signomial,RationalSignomial}, num_samples::Int=10;
               return_matrices=false, mode=OscarMode(), tol=1e-10)

Return a sampled Hoffman-constant lower bound for the stored expression of
`f`. With `return_matrices=true`, return `(bound, A, b)`.
"""
function lower_hoff(f::Union{Signomial, RationalSignomial},
        num_samples::Int = 10;
        return_matrices::Bool = false,
        mode::LinearRegionsCalculationMode = OscarMode(),
        tol::Float64 = 1e-10)
    A, b = linearmap_matrices(f; mode = mode)
    t_matrices, empty_return = _t_matrices_or_inf(A, b, return_matrices)
    empty_return !== nothing && return empty_return
    # The maximum of the per-matrix lower bounds is a lower bound for the
    # maximum of the per-matrix Hoffman constants.
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

# Effective-radius bounds.

@doc raw"""
    exact_er(f::Signomial; mode::LinearRegionsCalculationMode=OscarMode())

Return an infinity-norm effective-radius bound using [`exact_hoff`](@ref).
"""
function exact_er(f::Signomial; mode::LinearRegionsCalculationMode = OscarMode())
    hoff_const, A, b = exact_hoff(f, return_matrices = true, mode = mode)
    isinf(hoff_const) && return Inf
    tilde_bs = tilde_vectors(b)
    return hoff_const *
           maximum([norm(positive_component(tilde_b), Inf) for tilde_b in tilde_bs])
end

@doc raw"""
    upper_er(f::Signomial; mode::LinearRegionsCalculationMode=OscarMode())

Return an infinity-norm effective-radius bound using [`upper_hoff`](@ref).
"""
function upper_er(f::Signomial; mode::LinearRegionsCalculationMode = OscarMode())
    hoff_upper, A, b = upper_hoff(f, return_matrices = true, mode = mode)
    isinf(hoff_upper) && return Inf
    tilde_bs = tilde_vectors(b)
    return hoff_upper *
           maximum([norm(positive_component(tilde_b), Inf) for tilde_b in tilde_bs])
end

@doc raw"""
    exact_er(f::RationalSignomial; mode::LinearRegionsCalculationMode=OscarMode())

Return an infinity-norm effective-radius bound using [`exact_hoff`](@ref).
"""
function exact_er(f::RationalSignomial; mode::LinearRegionsCalculationMode = OscarMode())
    hoff_const, A, b = exact_hoff(f, return_matrices = true, mode = mode)
    isinf(hoff_const) && return Inf
    return hoff_const * max(maximum(b[1]) - minimum(b[1]), maximum(b[2]) - minimum(b[2]))
end

@doc raw"""
    upper_er(f::RationalSignomial; mode::LinearRegionsCalculationMode=OscarMode())

Return an infinity-norm effective-radius bound using [`upper_hoff`](@ref).
"""
function upper_er(f::RationalSignomial; mode::LinearRegionsCalculationMode = OscarMode())
    hoff_upper, A, b = upper_hoff(f, return_matrices = true, mode = mode)
    isinf(hoff_upper) && return Inf
    return hoff_upper * max(maximum(b[1]) - minimum(b[1]), maximum(b[2]) - minimum(b[2]))
end
