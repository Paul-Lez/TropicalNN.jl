# Remove monomials that do not affect a tropical function on an open set.

@doc raw"""
    prune(f::Signomial{T};
          parallel::Bool=true, workers=nothing,
          mode::LinearRegionsCalculationMode=OscarMode())

Return a copy of `f` without monomials whose dominance polyhedron is not
full-dimensional. `mode` selects the polyhedral backend. If `parallel=true`
and `workers` is an `AbstractWorkerPool`, the checks run on those processes.
"""
function prune(
        f::Signomial;
        parallel::Bool = true,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
        mode::LinearRegionsCalculationMode = OscarMode()
)
    keep = if parallel
        _strong_elim_keep_mask(f, workers, mode)
    else
        _strong_elim_keep_mask(f, nothing, mode)
    end

    return _filter_monomials(f, keep)
end

"""
    _filter_monomials(f, keep)

Select monomials from the input `Signomial` `f`. The Boolean vector `keep`
must have one value for each monomial. Return a `Signomial` that
contains the selected monomials.
"""
function _filter_monomials(f::Signomial{T}, keep::AbstractVector{Bool}) where {T}
    new_exp = Vector{Vector{T}}()
    sizehint!(new_exp, count(keep))
    new_coeff = Dict{Vector{T}, Oscar.TropicalSemiringElem{typeof(max)}}()

    for i in Base.eachindex(f)
        if keep[i]
            e = Vector{T}(get_exp(f, i))
            push!(new_exp, e)
            new_coeff[e] = get_coeff(f, i)
        end
    end

    return Signomial(new_coeff, new_exp)
end

"""
    _strong_elim_keep_mask(f, workers, mode)

Find the monomials of the input `Signomial` `f` that have full-dimensional
dominance regions. `workers` is `nothing` or an `AbstractWorkerPool`, and
`mode` selects the polyhedral backend. Return a Boolean vector in which `true`
marks a monomial with a full-dimensional dominance region.
"""
function _strong_elim_keep_mask(
        f::Signomial,
        workers::Union{Nothing, Distributed.AbstractWorkerPool},
        mode::LinearRegionsCalculationMode
)
    n = length(f)
    if workers === nothing || n <= 1
        return _strong_elim_keep_chunk((f, 1:n, mode))
    end

    _assert_tropicalnn_loaded(workers)
    chunks = _index_chunks(n, length(Distributed.workers(workers)))
    chunk_results = Distributed.pmap(
        _strong_elim_keep_chunk,
        workers,
        [(f, chunk, mode) for chunk in chunks]
    )
    return Base.reduce(vcat, chunk_results)
end

"""
    _strong_elim_keep_chunk(args)

Check one ordered collection of monomial indices. The input is a tuple `(f,
inds, mode)` with a `Signomial`, its indices to check, and a polyhedral backend
mode. Return a Boolean vector in the same order as `inds`; `true` marks a
monomial with a full-dimensional dominance region.
"""
function _strong_elim_keep_chunk(args)
    f, inds, mode = args
    keep = Vector{Bool}(undef, length(inds))
    # A discarded monomial does not affect later full-dimensionality checks.
    competitors = collect(Base.eachindex(f))
    for (j, i) in pairs(inds)
        poly = polyhedron(f, i, mode; competitors = competitors)
        keep[j] = is_full_dimensional(poly; mode = mode)
        if !keep[j]
            deleteat!(competitors, searchsortedfirst(competitors, i))
        end
    end
    return keep
end

@doc raw"""
    prune(f::RationalSignomial{T};
          parallel::Bool=true, workers=nothing,
          mode::LinearRegionsCalculationMode=OscarMode())

Return a copy of `f` with its numerator and denominator pruned independently.

# Arguments

- `f::RationalSignomial{T}`: Rational function to prune.
- `parallel=true`: Permit parallel checks.
- `workers=nothing`: Optional Julia worker pool.
- `mode=OscarMode()`: Backend for full-dimensionality checks.
"""
function prune(
        f::RationalSignomial;
        parallel::Bool = true,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return RationalSignomial(
        prune(f.num; parallel = parallel, workers = workers, mode = mode),
        prune(f.den; parallel = parallel, workers = workers, mode = mode)
    )
end

@doc raw"""
    prune(F::Vector{RationalSignomial{T}};
          parallel::Bool=true, workers=nothing,
          mode::LinearRegionsCalculationMode=OscarMode())

Prune each rational function in `F`.

# Arguments

- `F::Vector{RationalSignomial{T}}`: Rational functions to prune.
- `parallel=true`: Permit parallel checks.
- `workers=nothing`: Optional Julia worker pool.
- `mode=OscarMode()`: Backend for full-dimensionality checks.
"""
function prune(
        F::Vector{<:RationalSignomial};
        parallel::Bool = true,
        workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return [prune(f; parallel = parallel, workers = workers, mode = mode) for f in F]
end

Base.@deprecate reduce(
    f::Signomial;
    parallel::Bool = true,
    workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
    mode::LinearRegionsCalculationMode = OscarMode()
) prune(f; parallel = parallel, workers = workers, mode = mode) false

Base.@deprecate reduce(
    f::RationalSignomial;
    parallel::Bool = true,
    workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
    mode::LinearRegionsCalculationMode = OscarMode()
) prune(f; parallel = parallel, workers = workers, mode = mode) false

Base.@deprecate reduce(
    F::Vector{<:RationalSignomial};
    parallel::Bool = true,
    workers::Union{Nothing, Distributed.AbstractWorkerPool} = nothing,
    mode::LinearRegionsCalculationMode = OscarMode()
) prune(F; parallel = parallel, workers = workers, mode = mode) false
