# Matrix construction.

@doc raw"""
    _linearmap_matrices(f::Signomial; mode::LinearRegionsCalculationMode=OscarMode())

Return exponent matrix `A` and coefficient vector `b` for the non-redundant monomials of `f`.
"""
function _linearmap_matrices(f::Signomial; mode::LinearRegionsCalculationMode = OscarMode())
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
    _linearmap_matrices(f::RationalSignomial; mode::LinearRegionsCalculationMode=OscarMode())

Return the exponent matrices and coefficient vectors for the numerator and
denominator of `f`.
"""
function _linearmap_matrices(
        f::RationalSignomial;
        mode::LinearRegionsCalculationMode = OscarMode()
)
    Anum, bnum = _linearmap_matrices(f.num; mode = mode)
    Aden, bden = _linearmap_matrices(f.den; mode = mode)
    return (Anum, Aden), (bnum, bden)
end

@doc raw"""
    _tilde_matrices(A::Matrix)

Return `A - ones(m) * A[i, :]'` for each row `i` of `A`.
"""
function _tilde_matrices(A::Matrix)
    m, n = size(A)
    ones_vector = ones(m, 1)
    return [A - ones_vector * reshape(A[row, :], (1, n)) for row in 1:m]
end

@doc raw"""
    _tilde_matrices(As::Tuple{Matrix, Matrix})

Return the transformed matrices for all numerator-denominator row
pairs.
"""
function _tilde_matrices(As::Tuple{Matrix, Matrix})
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
    _tilde_vectors(b::Vector)

Return `b - b[i] * ones(length(b))` for each index `i`.
"""
function _tilde_vectors(b::Vector)
    return [b - b[row] * ones(length(b)) for row in 1:length(b)]
end

@doc raw"""
    _positive_component(b::Vector)

Return `max.(b, 0)` as a vector.
"""
function _positive_component(b::Vector)
    return vec([max(0, entry) for entry in b])
end

"""
    _hoff_with_matrices(matrix_hoff, f; mode)

Apply `matrix_hoff` to each transformed matrix for `f`. Return the result and
the exponent and coefficient data used in the calculation.
"""
function _hoff_with_matrices(
        matrix_hoff::Function,
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode
)
    A, b = _linearmap_matrices(f; mode = mode)
    t_matrices = _tilde_matrices(A)
    isempty(t_matrices) && return 0.0, A, b

    hoff_value = 0.0
    for tilde_matrix in t_matrices
        hoff_value = max(hoff_value, matrix_hoff(tilde_matrix))
    end
    return hoff_value, A, b
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
    _surjectivity_test(A::Matrix; tol=1e-10) -> (v, t)

Test A-surjectivity with the GLPK problem
`min ‖A'v‖₁` subject to `sum(v) = 1` and `v ≥ 0`. Return `(v, t)`; the rows
are A-surjective when `t > 0`. `tol` removes small floating-point results.
"""
function _surjectivity_test(A::Matrix; tol::Float64 = 1e-10)
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
    _brute_force_hoff(A::Matrix; tol=1e-10)

Compute the infinity-norm Hoffman constant by testing every nonempty row
subset. The enumeration is exhaustive, and GLPK uses floating-point
arithmetic.
"""
function _brute_force_hoff(A::Matrix; tol::Float64 = 1e-10)
    m = size(A, 1)
    H = 0.0
    for j in 1:m
        for subset in Combinatorics.combinations(1:m, j)
            AA = A[subset, :]
            _, t = _surjectivity_test(AA; tol = tol)
            if t > 0
                H = max(H, 1 / t)
            end
        end
    end
    return H
end

@doc raw"""
    _pvz_hoff(A::Matrix; return_certificates=false, tol=1e-10)

Compute the infinity-norm Hoffman constant with the Peña--Vera--Zuluaga algorithm.
With `return_certificates=true`, return `(H, F, I)` with the surjective sets
and obstruction supports. GLPK uses floating-point arithmetic.
"""
function _pvz_hoff(A::Matrix; return_certificates::Bool = false, tol::Float64 = 1e-10)
    m = size(A, 1)
    H = 0.0

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
        x, t = _surjectivity_test(A[J, :]; tol = tol)

        if t > 0
            # A surjective set certifies itself and every subset below it.
            push!(F, J)
            H = max(H, 1 / t)

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

    if return_certificates
        return H, F, I
    else
        return H
    end
end

@doc raw"""
    hoffman_constant(A::Matrix; brute_force=false, tol=1e-10)

Compute the infinity-norm Hoffman constant of `A`. By default, use the
Peña--Vera--Zuluaga algorithm. Set `brute_force=true` to test every nonempty
row subset instead. Both algorithms solve floating-point LPs with GLPK.
"""
function hoffman_constant(A::Matrix; brute_force::Bool = false, tol::Float64 = 1e-10)
    if brute_force
        return _brute_force_hoff(A; tol = tol)
    else
        return _pvz_hoff(A; tol = tol)
    end
end

@doc raw"""
    upper_hoffman_constant(A::Matrix)

Return the largest `sqrt(length(J)) / minimum(svdvals(A[J, :]))` over
nonempty full-rank row subsets `J`.
"""
function upper_hoffman_constant(A::Matrix)
    m, n = size(A)
    HU = 0.0
    for j in 1:m
        for subset in Combinatorics.combinations(1:m, j)
            AJ = A[subset, :]
            if rank(AJ) == min(j, n)
                p_J = minimum(svdvals(AJ))
                if p_J > 0
                    HU = max(HU, sqrt(length(subset)) / p_J)
                end
            end
        end
    end
    return HU
end

@doc raw"""
    lower_hoffman_constant(A::Matrix, num_samples::Int=10; tol=1e-10)

Return a lower bound from `num_samples` random nonempty row subsets. If
`num_samples >= 2^m`, compute the exact value with brute force.
"""
function lower_hoffman_constant(
        A::Matrix,
        num_samples::Int = 10;
        tol::Float64 = 1e-10
)
    m, n = size(A)
    HL = 0.0
    # Enumerate all nonempty subsets when sampling would require more tests.
    if num_samples >= 2^m
        return hoffman_constant(A; brute_force = true, tol = tol)
    else
        for i in 1:num_samples
            K = rand(1:m)
            J = sort(unique(rand(1:m, K)))
            AJ = A[J, :]
            x, t = _surjectivity_test(AJ; tol = tol)
            if t > 0
                HL = max(HL, 1 / t)
            end
        end
    end
    return HL
end

@doc raw"""
    hoffman_constant(f::Union{Signomial,RationalSignomial};
                     brute_force=false, mode=OscarMode(), tol=1e-10)

Compute the Hoffman constant for the stored expression of `f`. By default, use
the Peña--Vera--Zuluaga algorithm on every transformed matrix. Set
`brute_force=true` to use exhaustive subset enumeration.
"""
function hoffman_constant(f::Union{Signomial, RationalSignomial};
        brute_force::Bool = false,
        mode::LinearRegionsCalculationMode = OscarMode(),
        tol::Float64 = 1e-10)
    algorithm = brute_force ? _brute_force_hoff : _pvz_hoff
    hoff_const, _, _ = _hoff_with_matrices(
        matrix -> algorithm(matrix; tol = tol),
        f;
        mode = mode
    )
    return hoff_const
end

@doc raw"""
    upper_hoffman_constant(f::Union{Signomial,RationalSignomial}; mode=OscarMode())

Return a Hoffman-constant upper bound for the stored expression of `f`.
"""
function upper_hoffman_constant(
        f::Union{Signomial, RationalSignomial};
        mode::LinearRegionsCalculationMode = OscarMode()
)
    hoff_upper, _, _ = _hoff_with_matrices(upper_hoffman_constant, f; mode = mode)
    return hoff_upper
end

@doc raw"""
    lower_hoffman_constant(f::Union{Signomial,RationalSignomial},
                           num_samples::Int=10; mode=OscarMode(), tol=1e-10)

Return a sampled Hoffman-constant lower bound for the stored expression of
`f`.
"""
function lower_hoffman_constant(f::Union{Signomial, RationalSignomial},
        num_samples::Int = 10;
        mode::LinearRegionsCalculationMode = OscarMode(),
        tol::Float64 = 1e-10)
    hoff_lower, _, _ = _hoff_with_matrices(
        matrix -> lower_hoffman_constant(matrix, num_samples; tol = tol),
        f;
        mode = mode
    )
    return hoff_lower
end

# Effective-radius bounds.

@doc raw"""
    exact_er(f::Signomial; brute_force=false, mode=OscarMode())

Return an infinity-norm effective-radius bound using [`hoffman_constant`](@ref).
"""
function exact_er(f::Signomial;
        brute_force::Bool = false,
        mode::LinearRegionsCalculationMode = OscarMode())
    hoff_const, _, b = _hoff_with_matrices(
        matrix -> hoffman_constant(matrix; brute_force = brute_force),
        f;
        mode = mode
    )
    iszero(hoff_const) && return 0.0
    tilde_bs = _tilde_vectors(b)
    return hoff_const *
           maximum([norm(_positive_component(tilde_b), Inf) for tilde_b in tilde_bs])
end

@doc raw"""
    upper_er(f::Signomial; mode::LinearRegionsCalculationMode=OscarMode())

Return an infinity-norm effective-radius bound using
[`upper_hoffman_constant`](@ref).
"""
function upper_er(f::Signomial; mode::LinearRegionsCalculationMode = OscarMode())
    hoff_upper, _, b = _hoff_with_matrices(upper_hoffman_constant, f; mode = mode)
    iszero(hoff_upper) && return 0.0
    tilde_bs = _tilde_vectors(b)
    return hoff_upper *
           maximum([norm(_positive_component(tilde_b), Inf) for tilde_b in tilde_bs])
end

@doc raw"""
    exact_er(f::RationalSignomial; brute_force=false, mode=OscarMode())

Return an infinity-norm effective-radius bound using [`hoffman_constant`](@ref).
"""
function exact_er(f::RationalSignomial;
        brute_force::Bool = false,
        mode::LinearRegionsCalculationMode = OscarMode())
    hoff_const, _, b = _hoff_with_matrices(
        matrix -> hoffman_constant(matrix; brute_force = brute_force),
        f;
        mode = mode
    )
    iszero(hoff_const) && return 0.0
    return hoff_const * max(maximum(b[1]) - minimum(b[1]), maximum(b[2]) - minimum(b[2]))
end

@doc raw"""
    upper_er(f::RationalSignomial; mode::LinearRegionsCalculationMode=OscarMode())

Return an infinity-norm effective-radius bound using
[`upper_hoffman_constant`](@ref).
"""
function upper_er(f::RationalSignomial; mode::LinearRegionsCalculationMode = OscarMode())
    hoff_upper, _, b = _hoff_with_matrices(upper_hoffman_constant, f; mode = mode)
    iszero(hoff_upper) && return 0.0
    return hoff_upper * max(maximum(b[1]) - minimum(b[1]), maximum(b[2]) - minimum(b[2]))
end
