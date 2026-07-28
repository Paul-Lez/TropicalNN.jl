# Example: geometric quantities for a tropical polynomial
#
# The polynomial is f(x, y) = max(0, x, y). Its three linear regions are
# unbounded polyhedra. Each graph vertex represents one region.

using Graphs
using TropicalNN

f = Signomial(
    [0, 0, 0],
    [[0 // 1, 0 // 1], [1 // 1, 0 // 1], [0 // 1, 1 // 1]]
)

println("Boundedness flags:")
display(bounds(f))

println("\nRegion volumes:")
display(volumes(f))

println("\nConvex-piece counts:")
display(polyhedron_counts(f))

graph = get_graph(f)
println("\nRegion-adjacency graph:")
println("  vertices: ", Graphs.nv(graph))
println("  edges:    ", Graphs.ne(graph))

println("\nBoundary directions:")
display(edge_directions(f))

println("\nFinite boundary vertices:")
display(vertex_collection(f))
