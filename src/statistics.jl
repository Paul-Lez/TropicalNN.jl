############### Utilities ###############

@doc raw"""
    separate_components(linear_regions::Dict)

Separates linear regions into its disjoint components.
"""
function separate_components(linear_regions::Dict)
    region_components=Dict()
    for (linear_map,value) in linear_regions
        polys=value["polyhedra"]
        # indetifying which polyhedra in the region have a non-trivial intersection
        has_intersect=Dict()
        for (poly1, poly2) in combinations(polys,2)
            intesection = Oscar.intersect(poly1,poly2)
            has_intersect[(poly1, poly2)]=Oscar.is_feasible(intesection)
        end
        # using the intersection information we can determine whether the collection of polyhedra correspond to disjoint linear regions
        region_components[linear_map]=components(polys,has_intersect)
    end
    return region_components
end

@doc raw"""
    map_statistic(statistic,f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational},oscar::Bool=false)

Applies a statistic function to a tropical polynomial or tropical rational map by calling the statistic on the corresponding linear regions.
"""
function map_statistic(statistic,f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    # Obtain the unbounded matrix representations
    reps=m_reps(f)
    # Obtain the corresponding polyhedron representations
    polys=polyhedra_from_reps(reps,true)
    # Retrive the linear maps operating on the polyhedra
    linear_maps=get_linear_maps(f,reps["f_indices"])
    # Identify the disjoint linear regions
    linear_regions=separate_components(get_linear_regions(polys,linear_maps))
    # Apply the statistics to the linear regions
    return statistic(linear_regions)
end

############### Region ###############

# Interior point

@doc raw"""
    interior_points(polys::Array)

Returns interior points for each polyhedron in the collection.
"""
function interior_points(polys::Array)
    component_interiors=[]
    for poly in polys
        # Obtain an interior point for each polyhedron in the collection
        vertices=Oscar.vertices(poly)
        push!(component_interiors,sum(vertices)/length(vertices))
    end
    return component_interiors
end

@doc raw"""
    interior_points(linear_regions::Dict)

Returns interior points for the collection of polyhedron comprising linear regions.
"""
function interior_points(linear_regions::Dict)
    interior_points=Dict()
    # Iterate through each linear region and identify interior points
    for (linear_map,components) in linear_regions
        components_interiors=[]
        for polys in components
            push!(components_interior,interior_points(polys))
        end
        interior_points[linear_map]=components_interiors
    end
    return interior_points
end

@doc raw"""
    interior_points(linear_regions::Dict)

Returns interior points for the collection of polyhedron comprising linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function interior_points(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return map_statistic(interior_points,f)
end

# Bounded

@doc raw"""
    bounds(polys::Array)

Determines whether the polyhedra in a collection are bounded.
"""
function bounds(polys::Array)
    component_bounded=[]
    for poly in polys
        # Identify which polyhedra are bounded
        if Oscar.is_bounded(poly)
            push!(component_bounded,true)
        else
            push!(component_bounded,false)
        end
    end
    return component_bounded
end

@doc raw"""
    bounds(linear_regions::Dict)

Determines whether the polyhedra constituting linear regions are bounded.
"""
function bounds(linear_regions::Dict)
    bounded=Dict()
    # Iterate through the linear regions and note the polyhedra which are bounded
    for (linear_map,components) in linear_regions
        components_bounded=[]
        for polys in components
            push!(components_bounded,bounds(polys))
        end
        bounded[linear_map]=components_bounded
    end
    return bounded
end

@doc raw"""
    bounds(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Determines whether the polyhedra constituting the linear region of a tropical polynomial or a tropical rational map are bounded.
"""
function bounds(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return map_statistic(bounds,f)
end

# Volumes of regions

@doc raw"""
    volumes(polys::Array)

Finds the volumes of the polyhedra in a collection.
"""
function volumes(polys::Array)
    # Determine the bounded polyhedra
    bds=bounds(polys)
    vols=[]
    for (poly,bd) in zip(polys,bds)
        # Compute the volume of the bounded
        if bd
            push!(vols,Float64.(Oscar.volume(poly)))
        else
            push!(vols,Inf)
        end
    end
    return vols
end

@doc raw"""
    volumes(linear_regions::Dict)

Finds the volumes of the linear regions.
"""
function volumes(linear_regions::Dict)
    vols=Dict()
    # Iterate throguh the linear regions and compute their volume
    # by summing the volumes of the constituent polyhedra
    for (linear_map,components) in linear_regions
        components_vols=[]
        for polys in components
            # Take the sum of the volumes of the polyhedra
            # within the linear regions
            push!(components_vols,sum(volumes(polys)))
        end
        vols[linear_map]=components_vols
    end
    return vols
end

@doc raw"""
    volumes(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Finds the volumes of the linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function volumes(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return map_statistic(volumes,f)
end

# Number faces contributing to a linear region

@doc raw"""
    polyhedron_counts(linear_regions::Dict)

Returns the number of polyhedra in each linear region.
"""
function polyhedron_counts(linear_regions::Dict)
    # Count the number of polyhedra within each component for each linear map
    poly_counts=Dict(linear_map => [length(component) for component in components] for (linear_map,components) in linear_regions)
    return poly_counts
end

@doc raw"""
    polyhedron_counts(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Returns the number of polyhedra in each linear region of the tropical polynomial or tropical rational map.
"""
function polyhedron_counts(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return map_statistic(polyhedron_counts,f)
end

# Construct graph

@doc raw"""
    get_graph(linear_regions::Dict)

Constructs a graph from linear regions, where linear regions are connected if they share an edge.
"""
function get_graph(linear_regions::Dict)
    function connection(polys_1,polys_2)
        for poly_1 in polys_1
            ndim=Oscar.ambient_dim(poly_1)
            for poly_2 in polys_2
                intersection=Oscar.intersect(poly_1,poly_2)
                # Only consider intersections that are non-trivial
                # in the ambient dimension of the polyhedra
                if Oscar.dim(intersection)==ndim-1
                    return intersection
                end
            end
        end
        return nothing
    end
    # Collect the regions and their corresponding linear maps
    region_polys=[]
    linear_maps=[]
    for (linear_map,components) in linear_regions
        for component in components
            push!(region_polys,component)
            push!(linear_maps,linear_map)
        end
    end

    # Consider the graph with dictionary data such that we can populate
    # them with various measures
    g=MetaGraph(Graphs.Graph();label_type=Int,vertex_data_type=Dict,edge_data_type=Dict,graph_data=nothing)

    num_regions=length(region_polys)
    
    # Add the nodes to the graph
    for k in 1:num_regions
        # Populate the node data with an interior point 
        # and the volume of the corresponding region
        g[k]=Dict("interior_point"=>interior_points(region_polys[k]),"volume"=>volumes(region_polys[k]))
    end
    # Add the edges between regions that are connected
    for k in 1:(num_regions-1)
        for j in 1:length(region_polys[k+1:end])
            int=connection(region_polys[k],region_polys[k+1:end][j])
            if int!=nothing
                # Populate the node data with the linear maps 
                # and the intersection of the connected regions
                g[k,k+j]=Dict("linear maps" => [linear_maps[k],linear_maps[k+j]], "intersection" => int)
            end
        end
    end
    return g
end

@doc raw"""
    get_graph(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Constructs a graph of linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function get_graph(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return map_statistic(get_graph,f)
end

# Count edges

@doc raw"""
    edge_count(g::MetaGraph)

Counts the number of edges in a graph.
"""
function edge_count(g::MetaGraph)
    return Graphs.ne(g)
end

@doc raw"""
    edge_count(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Counts the number of edges in the graph constructed from the linear regions of the corresponding tropical polynomial or tropical rational map.
"""
function edge_count(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    # For one-dimensional polyhedra, there is no concept of
    # gradient for the intersection
    if nvars(f)==1
        error("For univariate tropical polynomials or tropical rational maps use the vertex functions instead.")
    end
    return Graphs.ne(get_graph(f))
end

# Get edge gradients

@doc raw"""
    edge_gradients(edge_attributes::Dict)

Identifies the gradients of the edges eminating from each vertex, along with providing the gradients of each unique edge.
"""
function edge_gradients(g::MetaGraph)
    # Each vertex can only have one outgoing edge of a certain gradient.
    # However, multiple edges can exist with the same gradient.
    # Here we collect the gradients and their multiplicities.
    function add_g(gs,vertex,grad)
        if haskey(gs,vertex)
            push!(gs[vertex],grad)
        else
            gs[vertex]=[grad]
        end
        return gs
    end
    # We want to obtain the gradients of the linear regions as a collective
    gs_full=[]
    # We want to collect the gradients of the outgoing edges for each node separately
    gs_with_source=Dict()
    for e in collect(edge_labels(g))
        e_int=g[e[1],e[2]]["intersection"]
        vs=Oscar.vertices(e_int)
        if length(vs)==1 # In this case the edge is an infinite ray
            ray=Oscar.rays(e_int)[1]
            grad=ray[2]/ray[1]
            # We add the edge to the singular source node
            gs_with_source=add_g(gs_with_source,vs[1],grad)
            push!(gs_full,grad)
        else # In this case the edge is bounded by vertices
            grad=(vs[1][2]-vs[2][2])/(vs[1][1]-vs[2][1])
            # We add the edge to each node
            gs_with_source=add_g(gs_with_source,vs[1],grad)
            gs_with_source=add_g(gs_with_source,vs[2],grad)
            push!(gs_full,grad)
        end
    end
    gs_with_source["full"]=gs_full
    return gs_with_source
end

@doc raw"""
    edge_gradients(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Identifies the gradients of the edges eminating from each vertex, along with providing the gradients of each unique edge, for the linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function edge_gradients(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    # For one-dimensional polyhedra, there is no concept of
    # gradient for the intersection
    if nvars(f)==1
        error("Not supported for univariate tropical polynomials or tropical rational maps.")
    end
    return edge_gradients(get_graph(f))
end

# Edge lengths

@doc raw"""
    edge_lengths(g::MetaGraph)

Calculate the lengths of the edges at the intersection of linear regions. Only returns the lengths of finite edges.
"""
function edge_lengths(g::MetaGraph)
    function add_l(ls,vertex,len)
        if haskey(ls,vertex)
            push!(ls[vertex],len)
        else
            ls[vertex]=[len]
        end
        return ls
    end
    # We want to obtain the lengths of the linear regions as a collective
    ls_full=[]
    # We want to collect the lengths of the outgoing edges for each node separately
    ls_with_source=Dict()
    for e in collect(edge_labels(g))
        e_int=g[e[1],e[2]]["intersection"]
        vs=Oscar.vertices(e_int)
        if length(vs)!=1 # We only want to consider finite edges
            len=sqrt(Float64((vs[1][1]-vs[2][1])^2+(vs[1][2]-vs[2][2])^2))
            # We add the edge to each node
            ls_with_source=add_l(ls_with_source,vs[1],len)
            ls_with_source=add_l(ls_with_source,vs[2],len)
            push!(ls_full,len)
        end
    end
    ls_with_source["full"]=ls_full
    return ls_with_source
end

@doc raw"""
    edge_lengths(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Calculates the lengths of the edges eminating from each vertex, along with providing the length of each unique edge, for the linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function edge_lengths(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    # For one-dimensional polyhedra, there is no concept of
    # gradient for the intersection
    if nvars(f)==1
        error("Not supported for univariate tropical polynomials or tropical rational maps.")
    end
    return edge_lengths(get_graph(f))
end

# Edge directions

@doc raw"""
    edge_directions(g::MetaGraph)

Calculate the direction vector of the edges at the intersection of linear regions.
"""
function edge_directions(g::MetaGraph)
    function add_d(ds,vertex,d)
        if haskey(ds,vertex)
            push!(ds[vertex],d)
        else
            ds[vertex]=[d]
        end
        return ds
    end
    # We want to obtain the directions of the edges of the linear regions as a collective
    # We canonically choose the vector representation that is positive in the first component
    ds_full=[]
    # We want to collect the directions of the outgoing edges for each node separately.
    # We canonically choose the vector representation as the normalised direction vector from the
    # source node to the destination node.
    ds_with_source=Dict()
    for e in collect(edge_labels(g))
        e_int=g[e[1],e[2]]["intersection"]
        vs=Oscar.vertices(e_int)
        if length(vs)==1 # In this case the edge is an infinite ray
            ray=Float64.(Oscar.rays(e_int)[1])
            d=[ray[1],ray[2]]
            d=d./(sqrt(d[1]^2+d[2]^2))
            # We add the edge to the singular source node
            ds_with_source=add_d(ds_with_source,vs[1],d)
            push!(ds_full,d.*(abs(d[1])/d[1]))
        else # In this case the edge is bounded by vertices
            # We add the edge to each node
            d1=Float64.([vs[2][1]-vs[1][1],vs[2][2]-vs[1][2]])
            d2=Float64.([vs[1][1]-vs[2][1],vs[1][2]-vs[2][2]])
            ds_with_source=add_d(ds_with_source,vs[1],d1./sqrt(d1[1]^2+d1[2]^2))
            ds_with_source=add_d(ds_with_source,vs[2],d2./sqrt(d2[1]^2+d2[2]^2))
            push!(ds_full,d1.*(abs(d1[1])/d1[1]))
        end
    end
    ds_with_source["full"]=ds_full
    return ds_with_source
end

@doc raw"""
    edge_directions(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Calculate the direction vector of the edges at the intersection of linear regions for the corresponding tropical polynomial or tropical rational map.
"""
function edge_directions(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    # For one-dimensional polyhedra, there is no concept of
    # a direction for the intersection
    if nvars(f)==1
        error("Not supported for univariate tropical polynomials or tropical rational maps.")
    end
    return edge_directions(get_graph(f))
end

# Collect vertices

@doc raw"""
    vertex_collection(g::MetaGraph)

Collects the vertices of the linear regions, along with their multiplicities, that is, how many regions share that vertex. 
"""
function vertex_collection(g::MetaGraph)
    # Here we collect all the vertices of the graph
    vs=reduce(vcat,[Oscar.vertices(g[e[1],e[2]]["intersection"]) for e in collect(edge_labels(g))])
    # We then count the multiplicities of each vertex byinterating through vs
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

@doc raw"""
    vertex_collection(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Collects the vertices of the linear regions corresponding to the tropical polynomial or tropical rational map, along with their multiplicities, that is, how many regions share that vertex. 
"""
function vertex_collection(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return vertex_collection(get_graph(f))
end

# Count vertices

@doc raw"""
    vertex_count(g::MetaGraph)

Counts the number of vertices in the linear regions from which the graph was obtained.
"""
function vertex_count(g::MetaGraph)
    return length(collect(keys(vertex_collection(g))))
end

@doc raw"""
    vertex_count(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})

Counts the number of vertices in the linear regions corresponding to the tropical polynomial or tropical rational map.
"""
function vertex_count(f::Union{TropicalPuiseuxPoly,TropicalPuiseuxRational})
    return vertex_count(get_graph(f))
end