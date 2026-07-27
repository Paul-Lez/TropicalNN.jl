using Distributed
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
            sorted = false,
        )

        regions = linear_regions(f; mode = HiGHSMode(), workers = pool)
        @test length(regions) == 3
        @test all(region -> TropicalNN.is_feasible(region[2]; mode = HiGHSMode()), regions)
    end
finally
    isempty(worker_ids) || rmprocs(worker_ids)
end
