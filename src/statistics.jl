############### Utilities ###############

@doc"""
    separate_components(linear_regions::Dict)

Separates linear regions into its disjoint components.
"""
function separate_components(linear_regions::Dict)
    region_components=Dict()
    for (linear_map,value) in linear_regions
        polys=value["polyhedra"]
        has_intersect = Dict()
        for (poly1, poly2) in combinations(polys, 2)
            intesection = Oscar.intersect(poly1, poly2)
            has_intersect[(poly1, poly2)] = Oscar.is_feasible(intesection)
        end
        region_components[linear_map]=components(polys,has_intersect)
    end
    return region_components
end

@doc"""
    get_statistic(statistic,f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational},oscar::Bool=false)

Applies a statistic function to a tropical polynomial or tropical rational map by calling the statistic on the corresponding linear regions.
"""
function get_statistic(statistic,f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    reps=m_reps(f)
    polys=polyhedra_from_reps(reps,true)
    linear_maps=get_linear_maps(f,reps["f_indices"])
    linear_regions=separate_components(get_linear_regions(polys,linear_maps))
    return statistic(linear_regions)
end

############### Region ###############

# Bounded

@doc"""
    region_bounds(linear_regions::Dict)

Determines whether the polyhedra constituting linear regions are bounded.
"""
function region_bounds(linear_regions::Dict)
    bounded=Dict()
    for (linear_map,components) in linear_regions
        components_bounded=[]
        for polys in components
            component_bounded=[]
            for poly in polys
                if Oscar.is_bounded(poly)
                    push!(component_bounded,true)
                else
                    push!(component_bounded,false)
                end
            end
            push!(components_bounded,component_bounded)
        end
        bounded[linear_map]=components_bounded
    end
    return bounded
end

@doc"""
    region_bounds(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Determines whether the polyhedra constituting the linear region of a tropical polynomial or a tropical rational map are bounded.
"""
function region_bounds(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return get_statistic(region_bounds,f)
end

# Volumes of regions

@doc"""
    region_volumes(linear_regions::Dict)

Finds the volumes of the linear regions.
"""
function region_volumes(linear_regions::Dict)
    bounds=region_bounds(linear_regions)
    vols=Dict()
    for linear_map in collect(keys(linear_regions))
        components_vols=[]
        for (c_ps,c_bs) in zip(linear_regions[linear_map],bounds[linear_map])
            component_vols=[]
            for (c_p,c_b) in zip(c_ps,c_bs)
                if c_b
                    push!(component_vols,Float64.(Oscar.volume(c_p)))
                else
                    push!(component_vols,Inf)
                end
            end
            push!(components_vols,sum(component_vols))
        end
        vols[linear_map]=components_vols
    end
    return vols
end

@doc"""
    region_volumes(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Finds the volumes of the linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function region_volumes(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return get_statistic(region_volumes,f)
end

# Number faces contributing to a linear region

@doc"""
    region_polyhedron_counts(linear_regions::Dict)

Returns the number of polyhedra in each linear region.
"""
function region_polyhedron_counts(linear_regions::Dict)
    p_counts=Dict()
    for (linear_map,components) in linear_regions
        p_counts[linear_map]=[length(component) for component in components]
    end
    return p_counts
end

@doc"""
    region_polyhedron_counts(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Returns the number of polyhedra in each linear region of the tropical polynomial or tropical rational map.
"""
function region_polyhedron_counts(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return get_statistic(region_polyhedron_counts,f)
end

@doc"""
    get_graph(linear_regions::Dict)

Constructs a graph from linear regions, where linear regions are connected if they share an edge.
"""
function get_graph(linear_regions::Dict)
    function connection(polys_1,polys_2)
        for poly_1 in polys_1
            ndim=Oscar.ambient_dim(poly_1)
            for poly_2 in polys_2
                intersection=Oscar.intersect(poly_1,poly_2)
                if Oscar.dim(intersection)==ndim-1
                    return intersection
                end
            end
        end
        return nothing
    end
    region_polys=[]
    linear_maps=[]
    for (linear_map,components) in linear_regions
        for component in components
            push!(region_polys,component)
            push!(linear_maps,linear_map)
        end
    end
    g=SimpleGraph(length(region_polys))
    edge_attributes=Dict()
    for k in 1:(length(region_polys)-1)
        for j in 1:length(region_polys[k+1:end])
            int=connection(region_polys[k],region_polys[k+1:end][j])
            if int!=nothing
                Graphs.add_edge!(g,k,k+j)
                edge_attributes[Graphs.Edge(k,k+j)]=Dict("linear maps" => [linear_maps[k],linear_maps[k+j]], "intersection" => int, "vertices" => Oscar.vertices(int))
            end
        end
    end
    return g,edge_attributes
end

@doc"""
    get_graph(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Constructs a graph of linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function get_graph(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return get_statistic(get_graph,f)
end

@doc"""
    edge_count(g::SimpleGraph)

Counts the number of edges in a graph.
"""
function edge_count(g::SimpleGraph)
    return Graphs.ne(g)
end

@doc"""
    edge_count(g::SimpleGraph)

Counts the number of edges in the graph constructed from the linear regions of the corresponding tropical polynomial or tropical rational map.
"""
function edge_count(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    if nvars(f)==1
        error("For univariate tropical polynomials or tropical rational maps use the vertex functions instead.")
    end
    return Graphs.ne(get_graph(f)[1])
end

@doc"""
    edge_gradients(edge_attributes::Dict)

Identifies the gradients of the edges eminating from each vertex, along with providing the gradients of each unique edge.
"""
function edge_gradients(edge_attributes::Dict)
    function add_g(gs,vertex,grad)
        if haskey(gs,vertex)
            push!(gs[vertex],grad)
        else
            gs[vertex]=[grad]
        end
        return gs
    end
    gs_full=[]
    gs_with_source=Dict()
    for attribute in collect(values(edge_attributes))
        vs=attribute["vertices"]
        if length(vs)==1
            ray=Oscar.rays(attribute["intersection"])[1]
            grad=ray[2]/ray[1]
            gs_with_source=add_g(gs_with_source,vs[1],grad)
            push!(gs_full,grad)
        else
            grad=(vs[1][2]-vs[2][2])/(vs[1][1]-vs[2][1])
            gs_with_source=add_g(gs_with_source,vs[1],grad)
            gs_with_source=add_g(gs_with_source,vs[2],grad)
            push!(gs_full,grad)
        end
    end
    gs_with_source["full"]=gs_full
    return gs_with_source
end

@doc"""
    edge_gradients(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Identifies the gradients of the edges eminating from each vertex, along with providing the gradients of each unique edge, for the linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function edge_gradients(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    if nvars(f)==1
        error("Not supported for univariate tropical polynomials or tropical rational maps.")
    end
    return edge_gradients(get_graph(f)[2])
end

@doc"""
    vertex_collection(edge_attributes::Dict)

Collects the vertices of the linear regions, along with their multiplicities, that is, how many regions share that vertex. 
"""
function vertex_collection(edge_attributes::Dict)
    vs=reduce(vcat,[v["vertices"] for v in collect(values(edge_attributes))])
    vs_with_mult=Dict()
    for v in vs
        if haskey(vs_with_mult,v)
            vs_with_mult[v]+=1
        else
            vs_with_mult[v]=1
        end
    end
    return vs_with_mult
end

@doc"""
    vertex_collection(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Collects the vertices of the linear regions corresponding to the tropical polynomial or tropical rational map, along with their multiplicities, that is, how many regions share that vertex. 
"""
function vertex_collection(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return vertex_collection(get_graph(f)[2])
end

@doc"""
    vertex_count(edge_attributes::Dict)

Counts the number of vertices in the linear regions from which the graph was obtained.
"""
function vertex_count(edge_attributes::Dict)
    return length(collect(keys(vertex_collection(edge_attributes))))
end

@doc"""
    vertex_count(edge_attributes::Dict)

Counts the number of vertices in the linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function vertex_count(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return vertex_count(get_graph(f)[2])
end