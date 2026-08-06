using BenchmarkTools
using TropicalNN

const Q = Rational{BigInt}
const SAMPLE_COUNT = parse(Int, get(ENV, "TROPICALNN_BENCH_SAMPLES", "10"))
const EVAL_COUNT = parse(Int, get(ENV, "TROPICALNN_BENCH_EVALS", "1"))

"""Return median time and allocation counts from a benchmark trial."""
function benchmark_summary(trial)
    estimate = BenchmarkTools.median(trial)
    return (time_ms = estimate.time / 1.0e6, allocations = estimate.allocs)
end

"""Benchmark global and layerwise workflows for one network."""
function run_case(name, W, b, thresholds; mode = HiGHSMode(), workers = nothing)
    # Warm both paths before measuring.
    global_expression = mlp_to_trop(W, b, thresholds)
    global_regions = linear_regions(global_expression; mode = mode, workers = workers)
    layerwise_regions, stage_stats = linear_regions(
        W,
        b,
        thresholds;
        mode = mode,
        workers = workers,
        return_stats = true
    )
    length(global_regions) == length(layerwise_regions) || error(
        "$name region-count mismatch: global=$(length(global_regions)), " *
        "layerwise=$(length(layerwise_regions))"
    )

    conversion_trial = @benchmark mlp_to_trop(
        $W,
        $b,
        $thresholds
    ) samples=SAMPLE_COUNT evals=EVAL_COUNT
    regions_trial = @benchmark linear_regions(
        $global_expression;
        mode = $mode,
        workers = $workers
    ) samples=SAMPLE_COUNT evals=EVAL_COUNT
    global_total_trial = @benchmark begin
        expression=mlp_to_trop($W, $b, $thresholds)
        linear_regions(expression; mode = $mode, workers = $workers)
    end samples=SAMPLE_COUNT evals=EVAL_COUNT
    layerwise_trial = @benchmark linear_regions(
        $W,
        $b,
        $thresholds;
        mode = $mode,
        workers = $workers
    ) samples=SAMPLE_COUNT evals=EVAL_COUNT

    numerator_monomials = sum(length(q.num) for q in global_expression)
    denominator_monomials = sum(length(q.den) for q in global_expression)
    result = (
        name = name,
        region_count = length(global_regions),
        numerator_monomials = numerator_monomials,
        denominator_monomials = denominator_monomials,
        global_conversion = benchmark_summary(conversion_trial),
        global_regions = benchmark_summary(regions_trial),
        global_total = benchmark_summary(global_total_trial),
        layerwise_total = benchmark_summary(layerwise_trial),
        layerwise_stages = stage_stats
    )
    display(result)
    return result
end

cases = [
    (
        "deep-1d",
        [Q.([1; -1;;]), Q.([1 1; 1 -1]), Q.([2 -1])],
        [Q.([0, 0]), Q.([0, 1]), Q.([0])],
        [Q.([0, 0]), Q.([-1, 0])]
    ),
    (
        "deep-2d",
        [Q.([1 0; 0 1]), Q.([1 -1; 1 1]), Q.([1 -2])],
        [Q.([0, -1]), Q.([0, 1]), Q.([0])],
        [Q.([0, 0]), Q.([-1, 0])]
    )
]

results = [run_case(case...) for case in cases]
