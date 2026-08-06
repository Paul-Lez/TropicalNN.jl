using Distributed
using Oscar
using Test
using TropicalNN

project_dir = normpath(joinpath(@__DIR__, ".."))
worker_ids = Int[]

try
    append!(worker_ids, addprocs(2; exeflags = ["--project=$project_dir"]))
    @everywhere using TropicalNN

    @testset "Distributed linear regions" begin
        pool = WorkerPool(worker_ids)
        @test length(Distributed.workers(pool)) == 2

        R = tropical_semiring(max)
        f = Signomial(
            [R(0), R(0), R(0)],
            [[0//1, 0//1], [1//1, 0//1], [0//1, 1//1]];
            sorted = false
        )

        regions = linear_regions(f; mode = HiGHSMode(), workers = pool)
        @test length(regions) == 3
        @test all(region -> TropicalNN.is_feasible(region[2]; mode = HiGHSMode()), regions)

        constant = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        max_x = Signomial([R(0), R(0)], [[0//1, 0//1], [1//1, 0//1]]; sorted = false)
        max_y = Signomial([R(0), R(0)], [[0//1, 0//1], [0//1, 1//1]]; sorted = false)
        q = [max_x / constant, max_y / constant]

        local_regions = linear_regions(q; mode = HiGHSMode())
        distributed_regions = linear_regions(q; mode = HiGHSMode(), workers = pool)
        region_signature(regions) = (
            length(regions), sort([length(region) for region in regions]))
        @test region_signature(distributed_regions) == (4, [1, 1, 1, 1])
        @test region_signature(distributed_regions) == region_signature(local_regions)

        scalar_local = linear_regions(max_x / constant; mode = HiGHSMode())
        scalar_distributed = linear_regions(max_x / constant; mode = HiGHSMode(), workers = pool)
        @test region_signature(scalar_distributed) == region_signature(scalar_local)

        Q = Rational{BigInt}
        W = [Q.([1 0; 0 1]), Q.([1 -1; 1 1]), Q.([1 -2])]
        b = [Q.([0, 0]), Q.([0, 1]), Q.([0])]
        thresholds = [Q.([0, 0]), Q.([-1, 0])]
        layerwise_local = linear_regions(
            W,
            b,
            thresholds;
            mode = HiGHSMode()
        )
        layerwise_distributed = linear_regions(
            W,
            b,
            thresholds;
            mode = HiGHSMode(),
            workers = pool
        )
        @test region_signature(layerwise_distributed) == region_signature(layerwise_local)
    end
finally
    isempty(worker_ids) || rmprocs(worker_ids)
end
