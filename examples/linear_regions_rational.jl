# Example: linear regions of a tropical Puiseux rational function
#
# We compute the linear regions of
#
#   q = max(x, y, x+y) - max(x, x+2y)
#
# represented as a RationalSignomial.  The linear regions are computed using
# HiGHSMode(), which represents each region internally by constraints for
# {x : Ax ≤ b}.

using TropicalNN

# --- Numerator: max(x, y, x+y) -----------------------------------------------
#   exponent [1, 0]  →  x
#   exponent [0, 1]  →  y
#   exponent [1, 1]  →  x + y

num_exps = [[1, 0], [0, 1], [1, 1]]
num_coeffs = [0, 0, 0]
f = Signomial(num_coeffs, num_exps)

# --- Denominator: max(x, x+2y) -----------------------------------------------
#   exponent [1, 0]  →  x
#   exponent [1, 2]  →  x + 2y

den_exps = [[1, 0], [1, 2]]
den_coeffs = [0, 0]
g = Signomial(den_coeffs, den_exps)

# --- Rational function -------------------------------------------------------

q = RationalSignomial(f, g)

# --- Compute linear regions via HiGHS ----------------------------------------

region_mode = HiGHSMode()
regions = linear_regions(q; mode = region_mode)

println("Tropical rational function:  max(x, y, x+y) - max(x, x+2y)")
println("Number of linear regions: ", length(regions))
println()

for (i, region) in enumerate(regions)
    # Each region is a LinearRegion containing one or more backend region objects.
    # A single convex piece has length 1; disconnected pieces have length > 1.
    if length(region) == 1
        A = get_matrix(region[1]; mode = region_mode)
        b = get_vector(region[1]; mode = region_mode)
        println("Region $i:  {x : Ax ≤ b}")
        println("  A = ", A)
        println("  b = ", b)
    else
        println("Region $i:  connected component of $(length(region)) polyhedra")
        for (k, piece) in enumerate(region)
            A = get_matrix(piece; mode = region_mode)
            b = get_vector(piece; mode = region_mode)
            println("  Piece $k:  A = ", A, ",  b = ", b)
        end
    end
end

# --- Hoffman constant of the rational function --------------------------------
# exact_hoff computes the exact value (brute force over row subsets);
# upper_hoff and lower_hoff give cheaper bounds.

hoff_exact = exact_hoff(q)
hoff_upper = upper_hoff(q)
hoff_lower = lower_hoff(q)

println()
println("Hoffman constant of the rational function:")
println("  exact:       ", hoff_exact)
println("  upper bound: ", hoff_upper)
println("  lower bound: ", hoff_lower)
