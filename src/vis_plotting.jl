# This file contains code to 
# generate the various plots.

@doc raw"""
    plotpoly(polyhedron,color,ax)

Plots a polyhedorn in a certain color on a given GLMakie axis.
"""
function plotpoly(poly,color,ax)
    if Polyhedra.fulldim(poly)==1
        # one-dimensional polyhedron are just lines
        vertices=[point[1] for point in collect(Polyhedra.points(poly))]
        GLMakie.lines!(ax,vertices,zeros(length(vertices)),color=color,linewidth=2)
    else
        # higher (two) dimensional polyhedra
        m=Polyhedra.Mesh(poly)
        try
            GLMakie.mesh!(ax,m,color=color,alpha=0.5)
        catch
        end
    end
end

@doc raw"""
    plotsurface(poly,linear_map,color,ax)

Given a polyhedron and its corresponding linear map, this function plots that the action of the linear map given the polyhedron as its input.
"""
function plotsurface(poly,linear_map,color,ax)
    input_vertices,output_vertices=get_surface_points(poly,linear_map)
    if length(input_vertices)==1
        GLMakie.lines!(ax,input_vertices[1],output_vertices,color=color,linewidth=2)
    else
        xs,ys=input_vertices[1],input_vertices[2]
        # the polyhedron is convex so we can plot the triangles formed by its vertices
        for triangle in Combinatorics.combinations(range(1,length(xs)),3)
            GLMakie.mesh!(ax,xs[triangle],ys[triangle],output_vertices[triangle],color=(color,0.9),overdraw=true,shading=NoShading)
        end
    end
end

@doc raw"""
    plotlevelset(f,poly,linear_map,level_set_value,ax)

Given a polyhedron, its corresponding linear map, and a the level set of focus, this function plots the region on the polyhedron where the level set is attained. 
"""
function plotlevelset(f,poly,linear_map,level_set_value,ax)
    component=get_level_set_component(polyhedra,linear_map,level_set_value)
    # if component is empty then the level set does not cross f in the region poly
    if length(component)>0
        if nvars(f)==1
            GLMakie.scatter!(ax,[point[1] for point in component],zeros(length(component)),color=:red)
        else
            GLMakie.lines!(ax,[point[1] for point in component],[point[2] for point in component],color=:red,linewidth=2)
        end
    end
end

@doc raw"""
    plot_linear_regions(f;bounding_box=nothing,rot_matrix=nothing,level_set_value=nothing)

Plots the linear regions of a tropical polynomial or tropical rational map, with each region color coordinated with respect to the linear map that operates on that region. If bounding box is not provided, then the regions are bounded by a region that encompases all the intersections between the polyhedra. A rotation matrix should be provided when `f` has more than two variables, such that the regions intersecting the rotated plane of the first two variables are plotted. If a level set value is provided, then the level set of f is depicted on the plot in red.
"""
function plot_linear_regions(f;bounding_box=nothing,rot_matrix=nothing,level_set_value=nothing)
    if nvars(f)>2 && rot_matrix==nothing
        error("Please supply a rotation matrix, even if it is the identity, when there are more than two input dimensions.")
    end
    fig=GLMakie.Figure()
    ax=GLMakie.Axis(fig[1,1])

    reps=m_reps(f,bounding_box,rot_matrix)
    polys=polyhedra_from_reps(reps)
    linear_maps=get_linear_maps(f,reps["f_indices"])
    linear_regions=get_linear_regions(polys,linear_maps)

    # plot the polyhedra for each linear region
    for key in collect(keys(linear_regions))
        region_color=linear_regions[key]["color"]
        region_polyhedra=linear_regions[key]["polyhedra"]
        for poly in region_polyhedra
            plotpoly(poly,region_color,ax)
        end
    end

    # if a level set value is specified we add the level set to the plot
    if level_set_value!=nothing
        for (linear_map,value) in linear_regions
            for poly in value["polyhedra"]
                plotlevelset(f,poly,linear_map,level_set_value,ax)
            end
        end
    end
    return fig,ax
end

@doc raw"""
    plot_linear_maps(f;bounding_box=nothing,xreversed=false,yreversed=false)

Plots the functional value of a tropical polynomial or tropical rational map, with the regions colored with respect to the polyhedra. If bounding box is not provided, then the input domain of the plot is bounded by a region that encompases all the intersections between the polyhedra. When `f` has two variables, xreversed and yreversed control the perspective of the resulting three-dimensional plot. This function is only support for tropical polynomials and tropical rational maps in at most two variables.
"""
function plot_linear_maps(f;bounding_box=nothing,xreversed=false,yreversed=false)
    if nvars(f)>2
        error("Only supported for rational maps with at most two input dimensions")
    end
    reps=m_reps(f,bounding_box)
    polys=polyhedra_from_reps(reps)
    linear_maps=get_linear_maps(f,reps["f_indices"])
    linear_regions=get_linear_regions(polys,linear_maps)

    fig=GLMakie.Figure()
    if nvars(f)==1
        ax=GLMakie.Axis(fig[1,1])
    else
        ax=GLMakie.Axis3(fig[1,1],xreversed=xreversed,yreversed=yreversed)
    end

    # plot the polyhedra for each linear region
    for (linear_map,value) in linear_regions
        region_color=value["color"]
        for poly in value["polyhedra"]
            plotsurface(poly,linear_map,region_color,ax)
        end
    end
    return fig,ax
end