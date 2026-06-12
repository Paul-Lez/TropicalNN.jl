# Example: full pipeline — MLP → tropical rational function → linear regions
#
# This script demonstrates the complete TropicalNN.jl workflow:
#   1. Construct a small ReLU MLP (architecture [2, 4, 1])
#   2. Convert it to an equivalent tropical Puiseux rational function
#   3. Enumerate its linear regions using the HiGHS LP backend
#   4. Report summary statistics on the regions
#
# Each linear region is a maximal connected subset of R² on which the
# network computes a single affine map.  The number of linear regions is
# a standard expressivity measure for ReLU networks.

using TropicalNN

# ---------------------------------------------------------------------------
# Step 1: Generate a random MLP with architecture [2, 4, 1]
# ---------------------------------------------------------------------------
# random_mlp returns (weights, biases, thresholds).
# By default thresholds are all zero (standard ReLU activation).
weights, biases, thresholds = random_mlp([2, 4, 1])

println("Network architecture: [2, 4, 1]")
println("  Layer 1 weight matrix size: ", size(weights[1]))
println("  Layer 2 weight matrix size: ", size(weights[2]))
println()

# ---------------------------------------------------------------------------
# Step 2: Convert to a tropical Puiseux rational function
# ---------------------------------------------------------------------------
# mlp_to_trop returns a Vector of RationalSignomial — one per output neuron.
# quicksum=true uses a faster polynomial addition (slight accuracy trade-off);
# strong_elim=true removes redundant monomials whose region is not full-dimensional.
tropical_funcs = mlp_to_trop(weights, biases, thresholds;
    quicksum = true, strong_elim = true)
f = tropical_funcs[1]   # single output neuron

println("Tropical rational function:")
println("  Numerator monomials:   ", monomial_count(f.num))
println("  Denominator monomials: ", monomial_count(f.den))
println()

# ---------------------------------------------------------------------------
# Step 3: Enumerate linear regions (HiGHS LP backend)
# ---------------------------------------------------------------------------
# HiGHSMode uses JuMP/HiGHS to check feasibility of each candidate region via
# LP. It returns a LinearRegions object whose elements are LinearRegion values,
# each holding one or more backend region objects encoding convex pieces.
region_mode = HiGHSMode()
regions = enum_linear_regions_rat_general(f; mode = region_mode)

println("Linear regions found: ", length(regions))
println()

# ---------------------------------------------------------------------------
# Step 4: Summary statistics
# ---------------------------------------------------------------------------
n_connected = sum(1 for r in regions if length(r) == 1)
n_disconnected = length(regions) - n_connected

println("Summary:")
println("  Regions with a single convex piece:        ", n_connected)
println("  Regions with multiple disconnected pieces: ", n_disconnected)
println()

# Print the first three regions for illustration
n_show = min(3, length(regions))
println("First $n_show region(s):")
for i in 1:n_show
    region = regions[i]
    if length(region) == 1
        A = get_matrix(region[1]; mode = region_mode)
        b = get_vector(region[1]; mode = region_mode)
        println("  Region $i  (1 convex piece, $(size(A, 1)) constraints)")
        println("    A = ", A)
        println("    b = ", b)
    else
        println("  Region $i  ($(length(region)) disconnected pieces)")
        for (k, piece) in enumerate(region)
            A = get_matrix(piece; mode = region_mode)
            println("    Piece $k: $(size(A, 1)) constraints")
        end
    end
    println()
end
