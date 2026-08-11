using Distributed
using Oscar: tropical_semiring
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
        @test all(region -> TropicalNN.is_feasible(only(region); mode = HiGHSMode()), regions)
        @test all(region -> only(region) isa Cell, regions)

        constant = Signomial([R(0)], [[0//1, 0//1]]; sorted = false)
        max_x = Signomial([R(0), R(0)], [[0//1, 0//1], [1//1, 0//1]]; sorted = false)
        max_y = Signomial([R(0), R(0)], [[0//1, 0//1], [0//1, 1//1]]; sorted = false)
        q = [max_x / constant, max_y / constant]

        region_signature(regions) = (
            length(regions), sort([length(region) for region in regions]))
        cell_signature(cell) = (
            Tuple(sort([
                (Tuple(cell.A[row, :]), cell.b[row]) for row in axes(cell.A, 1)
            ]; by = repr)),
            size(cell.matrix),
            Tuple(vec(cell.matrix)),
            Tuple(cell.offset)
        )
        region_data_signature(regions) = sort([
            sort(cell_signature.(collect(region)); by = repr) for region in regions
        ]; by = repr)

        for mode in (OscarMode(), HiGHSMode())
            local_regions = linear_regions(q; mode = mode)
            distributed_regions = linear_regions(q; mode = mode, workers = pool)
            @test region_signature(distributed_regions) == (4, [1, 1, 1, 1])
            @test region_data_signature(distributed_regions) ==
                  region_data_signature(local_regions)

            repeated_local = linear_regions(max_x / max_x; mode = mode)
            repeated_distributed = linear_regions(
                max_x / max_x; mode = mode, workers = pool)
            @test region_signature(repeated_distributed) == (1, [2])
            @test region_data_signature(repeated_distributed) ==
                  region_data_signature(repeated_local)

            layers = [q, q]
            layerwise_local = linear_regions(layers; mode = mode)
            layerwise_distributed = linear_regions(layers; mode = mode, workers = pool)
            @test region_data_signature(layerwise_distributed) ==
                  region_data_signature(layerwise_local)
        end
    end
finally
    isempty(worker_ids) || rmprocs(worker_ids)
end
