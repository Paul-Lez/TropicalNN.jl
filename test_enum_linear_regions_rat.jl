using TropicalNN
using Oscar

println("=" ^ 60)
println("Testing enum_linear_regions_rat")
println("=" ^ 60)

R = tropical_semiring(max)

# ---------------------------------------------------------------
# Bug 1: Docstring API mismatch
# The docstring (linear_regions.jl:129) shows the call as:
#   enum_linear_regions_rat(f::Signomial, g::Signomial, verbose)
# and the example uses:
#   enum_linear_regions_rat(f, g)
# But the actual signature is:
#   enum_linear_regions_rat(q::RationalSignomial)
# ---------------------------------------------------------------
println("\n--- Bug 1: Docstring API mismatch ---")
f_doc = Signomial([R(0), R(0)], [[1//1, 0//1], [0//1, 1//1]]; sorted=false)
g_doc = Signomial([R(0), R(0)], [[1//1, 1//1], [1//1, 2//1]]; sorted=false)

println("Calling enum_linear_regions_rat(f, g) as shown in the docstring...")
try
    result = enum_linear_regions_rat(f_doc, g_doc)
    println("UNEXPECTED SUCCESS: got $result")
catch e
    println("BUG CONFIRMED: Got $(typeof(e))")
    println("  Error: $e")
    println("  Users following the docstring example cannot call this function.")
end

println("\nCalling correctly with a RationalSignomial...")
q_doc = f_doc / g_doc
result_correct = enum_linear_regions_rat(q_doc)
println("SUCCESS: Got $(length(result_correct)) regions with correct call.")

# ---------------------------------------------------------------
# Bug 2: Inconsistent output types when linear maps repeat
#
# Construct a 2D example where:
#   f = max(x1+x2, x1, 1)  with exponents [1,1],[1,0],[0,0] and coeffs 0,0,1
#   g = max(x2, 0)          with exponents [0,1],[0,0]       and coeffs 0,0
#
# The pairs (f-exp=[1,1], g-exp=[0,1]) and (f-exp=[1,0], g-exp=[0,0])
# both yield the same linear map (exponent [1,0], coefficient 0, i.e. x1),
# but their intersection polyhedra are:
#   poly1 = { x2 >= 0, x1+x2 >= 1 }       (full-dimensional)
#   poly2 = { x2 <= 0, x1 >= 1 }           (full-dimensional)
#
# Since poly1 and poly2 share a boundary (x2=0, x1>=1 is feasible),
# they end up in the SAME component -> appended as [poly1, poly2].
#
# Meanwhile, two other pairs produce unique linear maps with single
# full-dimensional polyhedra, which are appended as bare polyhedra.
#
# Result: lin_regions mixes Oscar.Polyhedron and Vector{Oscar.Polyhedron}.
# ---------------------------------------------------------------
println("\n--- Bug 2: Output type inconsistency ---")

f2 = Signomial(
    Dict([1//1, 1//1] => R(0), [1//1, 0//1] => R(0), [0//1, 0//1] => R(1)),
    [[0//1, 0//1], [1//1, 0//1], [1//1, 1//1]]
)
g2 = Signomial(
    Dict([0//1, 1//1] => R(0), [0//1, 0//1] => R(0)),
    [[0//1, 0//1], [0//1, 1//1]]
)
q2 = f2 / g2

println("Calling enum_linear_regions_rat on a case with repeated linear maps...")
result2 = enum_linear_regions_rat(q2)
println("Got $(length(result2)) regions.")

types = [typeof(r) for r in result2]
unique_types = unique(types)

println("Element types in output:")
for (i, r) in enumerate(result2)
    println("  result[$(i)]: $(typeof(r))")
end

if length(unique_types) > 1
    println("\nBUG CONFIRMED: Output contains mixed types: $(unique_types)")
    println("  Some regions are bare polyhedra, others are arrays of polyhedra.")
    println("  This causes type errors if a caller treats all elements uniformly.")
    # Demonstrate the failure:
    println("\nAttempting Oscar.is_feasible on each element (as a user would)...")
    for (i, r) in enumerate(result2)
        try
            Oscar.is_feasible(r)
            println("  result[$i]: ok")
        catch e
            println("  result[$i]: ERROR - $(typeof(e)): $e")
        end
    end
else
    println("Types are consistent: $(unique_types[1])")
end

println("\n" * "=" ^ 60)
println("Summary of findings:")
println("  Bug 1 (API mismatch): enum_linear_regions_rat docstring shows")
println("    a 2-argument call but the function only accepts RationalSignomial.")
println("  Bug 2 (type inconsistency): when linear maps repeat, the output")
println("    mixes bare Oscar.Polyhedron and Vector{Oscar.Polyhedron},")
println("    making uniform iteration error-prone.")
println("=" ^ 60)
