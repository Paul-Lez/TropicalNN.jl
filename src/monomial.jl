# This file contains functions to convert a multilayer perceptron to a tropical Puiseux rational function, and to remove redundant monomials from the resulting function.

@doc raw"""
    reduce(f::Signomial{T};
           parallel::Bool=true, workers=nothing,
           mode::LinearRegionsCalculationMode=OscarMode())

Return a copy of `f` without monomials whose dominance polyhedron is not
full-dimensional. If `workers` is supplied and `parallel=true`, the
full-dimensionality checks run on those Julia worker processes. `mode`
selects the polyhedral backend used for the checks; use `OscarMode()` for exact
Oscar polyhedra or `HiGHSMode(threads=n)` for HiGHS LP checks with `n` solver
threads.
"""
function reduce(
        f::Signomial;
        parallel::Bool = true,
        workers = nothing,
        mode::LinearRegionsCalculationMode = OscarMode()
)
    keep = if parallel
        _strong_elim_keep_mask(f, workers, mode)
    else
        _strong_elim_keep_mask(f, nothing, mode)
    end

    return _filter_monomials(f, keep)
end

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

function _strong_elim_keep_mask(f::Signomial, workers, mode::LinearRegionsCalculationMode)
    n = length(f)
    worker_ids = _normalise_worker_ids(workers)
    if worker_ids === nothing || n <= 1
        return _strong_elim_keep_chunk((f, 1:n, mode))
    end

    _ensure_tropicalnn_loaded(worker_ids)
    chunks = _index_chunks(n, length(worker_ids))
    chunk_results = Distributed.pmap(
        _strong_elim_keep_chunk,
        Distributed.WorkerPool(worker_ids),
        [(f, chunk, mode) for chunk in chunks],
    )
    return Base.reduce(vcat, chunk_results)
end

function _strong_elim_keep_chunk(args)
    f, inds, mode = args
    keep = Vector{Bool}(undef, length(inds))
    for (j, i) in pairs(inds)
        poly = polyhedron(f, i, mode)
        keep[j] = is_full_dimensional(poly; mode = mode)
    end
    return keep
end

@doc raw"""
    reduce(f::RationalSignomial{T};
           parallel::Bool=true, workers=nothing,
           mode::LinearRegionsCalculationMode=OscarMode())

Removes redundant monomials from both numerator and denominator of a tropical
Puiseux rational function.

# Arguments
- `f::RationalSignomial{T}`: The rational function to simplify
- `parallel::Bool=true`: Whether to use process-parallel computation when
  workers are supplied
- `workers=nothing`: Optional Julia worker process ids
- `mode=OscarMode()`: Polyhedral backend used for full-dimensionality checks.
  Use `HiGHSMode(threads=n)` to run HiGHS checks with `n` solver threads.
"""
function reduce(
        f::RationalSignomial;
        parallel::Bool = true,
        workers = nothing,
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return RationalSignomial(
        reduce(f.num; parallel = parallel, workers = workers, mode = mode),
        reduce(f.den; parallel = parallel, workers = workers, mode = mode)
    )
end

@doc raw"""
    reduce(F::Vector{RationalSignomial{T}};
           parallel::Bool=true, workers=nothing,
           mode::LinearRegionsCalculationMode=OscarMode())

Removes redundant monomials from a vector of tropical Puiseux rational functions.

# Arguments
- `F::Vector{RationalSignomial{T}}`: The vector of rational functions to simplify
- `parallel::Bool=true`: Whether to use process-parallel computation when
  workers are supplied
- `workers=nothing`: Optional Julia worker process ids
- `mode=OscarMode()`: Polyhedral backend used for full-dimensionality checks.
  Use `HiGHSMode(threads=n)` to run HiGHS checks with `n` solver threads.
"""
function reduce(
        F::Vector{<:RationalSignomial};
        parallel::Bool = true,
        workers = nothing,
        mode::LinearRegionsCalculationMode = OscarMode()
)
    return [reduce(f; parallel = parallel, workers = workers, mode = mode) for f in F]
end

"""
    mlp_to_trop_with_strong_elim(linear_maps, bias, thresholds)

Deprecated; use `mlp_to_trop(linear_maps, bias, thresholds, strong_elim=true, dedup=true)`.
"""
function mlp_to_trop_with_strong_elim(linear_maps::Vector{Matrix{T}}, bias,
        thresholds) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    @warn "mlp_to_trop_with_strong_elim is deprecated, use mlp_to_trop(..., strong_elim=true, dedup=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, strong_elim = true, dedup = true)
end

"""
    mlp_to_trop_with_quicksum_with_strong_elim(linear_maps, bias, thresholds)

Deprecated; use `mlp_to_trop(linear_maps, bias, thresholds, quicksum=true, strong_elim=true)`.
"""
function mlp_to_trop_with_quicksum_with_strong_elim(linear_maps::Vector{Matrix{T}}, bias,
        thresholds) where {T <: Union{Oscar.scalar_types, Rational{BigInt}}}
    @warn "mlp_to_trop_with_quicksum_with_strong_elim is deprecated, use mlp_to_trop(..., quicksum=true, strong_elim=true) instead" maxlog=1
    return mlp_to_trop(linear_maps, bias, thresholds, quicksum = true, strong_elim = true)
end
