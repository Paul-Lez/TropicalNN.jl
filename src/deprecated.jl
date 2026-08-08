# Deprecated public bindings.

"""
    _warn_ignored_hoffman_mode(mode, function_name)

Warn that `mode` no longer controls a Hoffman calculation. Do nothing if
`mode` is `nothing`.
"""
function _warn_ignored_hoffman_mode(
        mode::Union{Nothing, LinearRegionsCalculationMode},
        function_name::Symbol
)
    isnothing(mode) && return nothing
    Base.depwarn(
        "`mode` is deprecated and does not affect this calculation. Call `prune(f; mode = mode)` first to remove redundant monomials.",
        function_name
    )
    return nothing
end

"""
    comp_with_quicksum(f, G)

Deprecated. Use `comp(f, G; quicksum=true)`.
"""
function comp_with_quicksum(f, G)
    Base.depwarn(
        "`comp_with_quicksum(f, G)` is deprecated. Use `comp(f, G; quicksum = true)`.",
        :comp_with_quicksum
    )
    return comp(f, G; quicksum = true)
end

Base.@deprecate mlp_to_trop tropicalize
