# Example: linear regions of a tropical Puiseux rational function
#
# We compute the linear regions of
#
#   q = max(x, y, x+y) - max(x-y, x+2y)
#
# represented as a TropicalPuiseuxRational.  The linear regions are computed
# using the HiGHS-based algorithm (enum_linear_regions_rat_highs), which
# represents each region as an (A, b) pair with {x : Ax ≤ b}.

using TropicalNN

# --- Numerator: max(x, y, x+y) -----------------------------------------------
#   exponent [1, 0]  →  x
#   exponent [0, 1]  →  y
#   exponent [1, 1]  →  x + y

num_exps   = [[1, 0], [0, 1], [1, 1]]
num_coeffs = [0, 0, 0]
f = TropicalPuiseuxPoly(num_coeffs, num_exps, false)

# --- Denominator: max(x-y, x+2y) ---------------------------------------------
#   exponent [1, -1]  →  x - y
#   exponent [1,  2]  →  x + 2y

den_exps   = [[1, -1], [1, 2]]
den_coeffs = [0, 0]
g = TropicalPuiseuxPoly(den_coeffs, den_exps, false)

# --- Rational function -------------------------------------------------------

q = TropicalPuiseuxRational(f, g)

# --- Compute linear regions via HiGHS ----------------------------------------

regions = enum_linear_regions_rat_highs(q)

println("Tropical rational function:  max(x, y, x+y) - max(x-y, x+2y)")
println("Number of linear regions: ", length(regions))
println()

for (i, region) in enumerate(regions)
    # A region is either a single (A, b) pair or a connected component
    # (array of (A, b) pairs sharing the same linear map).
    if region isa Tuple
        A, b = region
        println("Region $i:  {x : Ax ≤ b}")
        println("  A = ", A)
        println("  b = ", b)
    else
        println("Region $i:  connected component of $(length(region)) polyhedra")
        for (k, (A, b)) in enumerate(region)
            println("  Piece $k:  A = ", A, ",  b = ", b)
        end
    end
end
