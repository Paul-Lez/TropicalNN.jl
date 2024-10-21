function plotpoly(polyhedron,color,ax=nothing)

    if Polyhedra.fulldim(polyhedron)==1
        vertices=[point[1] for point in collect(Polyhedra.points(polyhedron))]
        GLMakie.lines!(vertices,zeros(length(vertices)),color=color,linewidth=2)
    else
        m=Polyhedra.Mesh(polyhedron)
        try
            GLMakie.mesh!(ax,m,color=color,alpha=0.5)
        catch
        end
    end
end

function plotsurface(poly,linear_map,color,ax)
    input_vertices,output_vertices=get_surface_points(poly,linear_map)
    if length(input_vertices)==1
        GLMakie.lines!(ax,input_vertices[1],output_vertices,color=color,linewidth=2)
    else
        xs,ys=input_vertices[1],input_vertices[2]
        for triangle in Combinatorics.combinations(range(1,length(xs)),3)
            GLMakie.mesh!(ax,xs[triangle],ys[triangle],output_vertices[triangle],color=(color,0.9),overdraw=true,shading=NoShading)
        end
    end
end

function plotlevelset(f,level_set_value,polyhedra,linear_maps,ax)
    components=get_level_set(level_set_value,polyhedra,linear_maps)
    if nvars(f)==1
        for component in components
            if length(component)>0
                GLMakie.scatter!(ax,[point[1] for point in component],zeros(length(component)),color=:red)
            end
        end
    else
        for component in components
            if length(component)>0
                GLMakie.lines!(ax,[point[1] for point in component],[point[2] for point in component],color=:red,linewidth=2)
            end
        end
    end
end

function plot_linear_regions(f;bounding_box=nothing,rot_matrix=nothing,level_set_value=nothing)
    if nvars(f)>2 && rot_matrix==nothing
        error("Please supply a rotation matrix, even if it is the identity, when there are more than two input dimensions.")
    end
    fig=GLMakie.Figure()
    ax=GLMakie.Axis(fig[1,1])

    reps=m_reps(f,bounding_box,rot_matrix)
    polys=polyhedra_from_reps(reps)
    linear_maps=get_linear_maps(f,reps["f_indices"])
    linear_regions=get_linear_regions(linear_maps,polys)

    for key in collect(keys(linear_regions))
        region_color=linear_regions[key]["color"]
        region_polyhedra=linear_regions[key]["polyhedra"]
        for poly in region_polyhedra
            plotpoly(poly,region_color,ax)
        end
    end

    if level_set_value!=nothing
        plotlevelset(f,level_set_value,polys,linear_maps,ax)
    end
    return fig,ax
end

function plot_linear_maps(f;bounding_box=nothing,xreversed=false,yreversed=false)
    if nvars(f)>2
        error("Only supported for rational maps with at most two input dimensions")
    end
    reps=m_reps(f,bounding_box)
    polys=polyhedra_from_reps(reps)
    linear_maps=get_linear_maps(f,reps["f_indices"])
    linear_regions=get_linear_regions(linear_maps,polys)

    fig=GLMakie.Figure()
    if nvars(f)==1
        ax=GLMakie.Axis(fig[1,1])
    else
        ax=GLMakie.Axis3(fig[1,1],xreversed=xreversed,yreversed=yreversed)
    end

    for (linear_map,value) in linear_regions
        region_color=value["color"]
        for poly in value["polyhedra"]
            plotsurface(poly,linear_map,region_color,ax)
        end
    end
    return fig,ax
end