# Deprecated public bindings.

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

Base.@deprecate Signomial_const signomial_const
Base.@deprecate Signomial_zero signomial_zero
Base.@deprecate Signomial_one signomial_one
Base.@deprecate SignomialMonomial signomial_monomial
Base.@deprecate RationalSignomial_identity rational_signomial_identity
Base.@deprecate RationalSignomial_zero rational_signomial_zero
Base.@deprecate RationalSignomial_one rational_signomial_one
Base.@deprecate mlp_to_trop tropicalize
