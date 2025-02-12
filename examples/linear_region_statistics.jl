using TropicalNN, GraphMakie, CairoMakie, Graphs

# Generate a random neural network and obtain its tropical representation
w,b,t=random_mlp([2,4,1])
f=mlp_to_trop_with_quicksum_with_strong_elim(w,b,t)[1]

# Our statistics are derived from the polyhedra representing the linear regions of the neural network.
# For simplicitly we compute these before hand such that they can used to derive multiple statistics.
# However, one could equally pass the tropical representation into these functions and obtain the same results.

# Here we obtain the matrix representation of the polyhedra
reps=m_reps(f)
# Here we get their polyehdron representation from the Oscar library
polys=polyhedra_from_reps(reps,true)
# Here we partition the regions according to the linear maps that operate on the regions
linear_maps=get_linear_maps(f,reps["f_indices"])
# Here we identify when regions with the same linear map are disconnected
linear_regions=separate_components(get_linear_regions(polys,linear_maps))

# Using the Oscar library we can obtain a few statistics on the geometry of these linear regions
# We can determine whether regions are bounded
bds=bounds(linear_regions)
println("Region Bounds:")
for (key,value) in bds
    println(string(value))
end
# We can determine the volume of these regions
vols=volumes(linear_regions)
println("Region Volumes:")
for (key,value) in vols
    println(string(value))
end
# We can determine how many polyhedra contribute to each linear region
counts=polyhedron_counts(linear_regions)
println("Region Polyhedra Counts:")
for (key,value) in counts
    println(string(value))
end

# We can also leverage the tropical representation to construct a graph on thte input space of this neural network.
# Namely, we take each linear region to be a node and add an edge between nodes when the linear regions are connected along a face.
G=get_graph(f)
# We can visualise this graph and compare it to the visualisation of the neural networks linear regions.
fig=GLMakie.Figure()
ax=Axis(fig[1,1])
graphplot!(ax,G)
GLMakie.save("./examples/outputs/statistics_graph.png",fig)
fig,ax=plot_linear_regions(f)
GLMakie.save("./examples/outputs/statistics_linear_regions.png",fig)

# We can also verify that the node data collected in the graph makes sense
scatter!(ax,reduce(vcat,[[Float64(p[1]) for p in G[v]["interior_point"]] for v in labels(G)]),reduce(vcat,[[Float64(p[2]) for p in G[v]["interior_point"]] for v in labels(G)]),color=:red)
text!(ax,reduce(vcat,[[Float64(p[1]) for p in G[v]["interior_point"]] for v in labels(G)]),reduce(vcat,[[Float64(p[2]) for p in G[v]["interior_point"]] for v in labels(G)]),text=reduce(vcat,[[string(vol) for vol in G[v]["volume"]] for v in labels(G)]))
GLMakie.save("./package/examples/outputs/statistics_linear_regions_with_data.png",fig)

# We use Graphs.jl to construct this graph and hence we can leverage its functionality to explore the properties of this graph.
# Some measures of interest may include the following.
# We can calculate the degree distribution of the graph
degree_distribution=Graphs.degree_histogram(G)
println("Degree Distribution: "*string(degree_distribution))
# We can calculate the density of the graph
density=Graphs.density(G)
println("Density: "*string(density))
# We can calculate the betweeness centrality of the nodes (linear regions) of the graph
betweenness_centrality=Graphs.betweenness_centrality(G)
println("Betweenness Centrality: "*string(betweenness_centrality))
# We can identify a vertex cover of the graph
vertex_cover=Graphs.vertex_cover(G, RandomVertexCover())
println("Size of Vertex Cover: "*string(length(vertex_cover)))
# We can identify a dominating set of the graph
dominating_set=Graphs.dominating_set(G,MinimalDominatingSet())
println("Size of Dominating Set: "*string(length(dominating_set)))