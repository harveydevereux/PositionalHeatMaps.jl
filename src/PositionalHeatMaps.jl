module PositionalHeatMaps
using ProgressMeter
using Plots
using PyCall
using Statistics
@pyimport scipy.interpolate as si
# TODO abstract to a box type? Is this needed?
# TODO N-dimensional?
export
Grid2D,InBox,PointsInBox,PlotBoxes,AssignBoxValues,HeatMap,BoxCentres,
AssignBoxVectors,Scaling,HeatMapLimits

# same as python's meshgrid
meshgrid(x,y) = (repeat(x',length(y),1),repeat(y,1,length(x)))

function Grid2D(points; n_boxes=100, bounds = zeros(4))
    """
        For a set of points (points) construct a 2D grid covering
        the entire area, by default, or passed bounds (bounds) with
        (n_boxes)
    """
    if bounds == zeros(4)
        min_x = minimum(points[:,1])
        max_x = maximum(points[:,1])
        min_y = minimum(points[:,2])
        max_y = maximum(points[:,2])
    else
        min_x,max_x,min_y,max_y = bounds[1],bounds[2],bounds[3],bounds[4]
    end

    # box length
    dx = (max_x-min_x)/n_boxes
    dy = (max_y-min_y)/n_boxes

    x = collect(min_x:dx:(max_x-dx))
    y = collect(min_y:dy:(max_y-dy))

    boxes = zeros((n_boxes^2,2,2))

    # create list of boxes
    k = 1
    for i in x
        for j in x
            boxes[k,:,:] = [i i+dx ; j j+dy]
            k += 1
        end
    end

    # return the boxes and their x,y coordinate for the "bottom left" point
    return boxes,x,y
end

function InBox(B, point)
    """
    Checks if point is in box B (Dx2 vector)

    A box is a set of upper and lower bounds in D dimensional space
    something like [l_1,u_1]x[l_2,u_2]x...x[l_D,u_D]

    This is then a Dx2 vector of bounds
    """
    if size(B,1) != size(point,1)
        "Need same dimension got"
        return false
    else
        for i in collect(1:size(B,1))
            if point[i] < B[i,1] || point[i] > B[i,2]
                return false
            end
        end
        return true
    end
end

function PointsInBox(B, X)
    """
    Wraps InBox to determine which of the nx2 matrix of
    points X is in box B

    A box is a set of upper and lower bounds in D dimensional space
    something like [l_1,u_1]x[l_2,u_2]x...x[l_D,u_D]

    This is then a Dx2 vector of bounds
    """
    In = []
    for i in collect(1:size(X,1))
        if InBox(B,X[i,:])
            push!(In,i)
        end
    end
    return In
end

function AssignBoxValues(points,boxes,values)
    """
        Takes a set of point, boxes, and associated values. Then for
        each box finds the points in it and assigns the mean of the
        associated values.

        points  Nx2
        boxes   Bx4
        values  Nx1
    """
    mean_box_values = zeros(size(boxes,1))
    @showprogress for i in 1:size(boxes,1)
        ind = PointsInBox(boxes[i,:,:],points[:,1:2])
        # ignore for no points in this box
        if isempty(points[ind,:])
            continue
        else
            mean_box_values[i] = mean(values[ind])
        end
    end
    return mean_box_values
end

function AssignBoxVectors(points,boxes,vectors)
    """
        Takes a set of point, boxes, and associated vectors. Then for
        each box finds the points in it and assigns the mean of the
        associated vectors (i.e a vector of the mean of each dimension).

        points  Nx2
        boxes   Bx4
        values  NxD
    """
    mean_box_vectors = zeros(size(boxes,1),size(vectors,2))
    @showprogress for i in 1:size(boxes,1)
        ind = PointsInBox(boxes[i,:,:],points[:,1:2])
        # ignore for no points in this box
        if isempty(points[ind,:])
            continue
        else
            mean_box_vectors[i,:] = mean(vectors[ind,:],dims=1)
        end
    end
    return mean_box_vectors
end

function BoxCentres(Boxes)
    """
        Returns the centre of each 2D box
    """
    C = zeros(size(Boxes,1),2)
    for i in 1:size(Boxes,1)
        C[i,:] = [abs(Boxes[i,1,1]-Boxes[i,1,2]), abs(Boxes[i,2,1]-Boxes[i,2,2])]
        C[i,:] = C[i,:]./2 .+ [Boxes[i,1,1],Boxes[i,2,1]]
    end
    return C
end

# scaling is useful to plot over the heatmaps with quiver plots for example.
function Scaling(coords,lim)
    """
        Min max scaling to the VoronoiDelaunay range requirement
    """
    Rescale = zeros(size(coords))
    for d in 1:2
        min = minimum(coords[:,d])
        max = maximum(coords[:,d])
        a = minimum(lim[d,:])
        b = maximum(lim[d,:])
        Rescale[:,d] = [ ((coords[i,d] - min)/(max-min))*(b-a)+a for i in 1:size(coords,1)]
    end
    return Rescale
end

function HeatMapLimits(Map)
    lims = zeros(2)
    for i in 1:size(Map,1)
        if isnan(Map[i,1])
            lims[1] = i
            break
        end
    end
    for j in 1:size(Map,2)
        if isnan(Map[1,j])
            lims[2] = j
            break
        end
    end
    return lims
end

function HeatMap(points,n_boxes,values; bounds=[-1000,1000,-1000,1000],interpolation="None",step=2)
    """
        Wraps around the whole module to produce a heatmap which
        is displayed and returned as a matrix.

        points          Nx2 coordinates
        n_boxes         number of boxes to approximate with
        values          Nx1 the desired value associated with each coordinates
        bounds          limits on which points to consider
        interpolation   whether to interpolate with python's scipy.interpolate.griddata
                        can do linear,cubic, or nearest (see the scipy page)
        step            the interpolation step
    """
    boxes,x,y = Grid2D(points,n_boxes=n_boxes,bounds=bounds)
    mean_box_values = AssignBoxValues(points,boxes,values)

    map = zeros(n_boxes,n_boxes)
    points = zeros(size(mean_box_values,1),2)
    k = 1
    for i in 1:size(x,1)
        for j in 1:size(y,1)
            map[i,j] = mean_box_values[k]
            points[k,:] = [x[i],y[j]]
            k += 1
        end
    end
    if (interpolation == "linear")
        xi = collect(bounds[1]:step:bounds[2])
        yi = collect(bounds[3]:step:bounds[4])
        xi,yi = meshgrid(xi,yi)
        G = si.griddata(points,mean_box_values,(xi,yi),method="linear")
        display(heatmap(G))
        return G,boxes
    end
    if (interpolation == "cubic")
        xi = collect(bounds[1]:step:bounds[2])
        yi = collect(bounds[3]:step:bounds[4])
        xi,yi = meshgrid(xi,yi)
        G = si.griddata(points,mean_box_values,(xi,yi),method="cubic")
        display(heatmap(G))
        return G,boxes
    end
    if (interpolation == "nearest")
        xi = collect(bounds[1]:step:bounds[2])
        yi = collect(bounds[3]:step:bounds[4])
        xi,yi = meshgrid(xi,yi)
        G = si.griddata(points,mean_box_values,(xi,yi),method="nearest")
        display(heatmap(G))
        return G,boxes
    end

    display(heatmap(map))
    return map,boxes
end

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

function PlotBoxes(boxes)
    plot()
    for i in 1:size(boxes,1)
        w = boxes[i,1,2] - boxes[i,1,1]
        h = boxes[i,2,2] - boxes[i,2,1]
        p = plot!(rectangle(w,h,boxes[i,1,1],boxes[i,2,1]));
    end
    plot!()
end

end # module gridded
