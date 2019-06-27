module PositionalHeatMaps
using ProgressMeter
using Plots
using PyCall
using Statistics
@pyimport scipy.interpolate as si
export ResponseMap

function ResponseMap(X,Y,Z;n_bins=100,limits=nothing,fill_value=0.0,square=true,exclude=-Inf,angleX=false,angleY=false,vectors=nothing)
    # handle odd bin size
    n_bins % 2 != 0 ? n_bins+=1 : nothing
    # filter with limits if using them
    if limits != nothing
        ind = ((X.>limits[1]).+(X.<limits[2]) .+ (Y.>limits[3]) .+ (Y.<limits[4])).==4
        X = X[ind]
        Y = Y[ind]
    end
    # remove some NaN values and (also remove xi if yi is NaN etc)
    ind = ((isnan.(X).==false) .+ (isnan.(Y).==false)) .== 2
    X = X[ind]
    Y = Y[ind]
    # bounds for indexing functions
    if square
        m = min(minimum(X),minimum(Y))
        M = max(maximum(X),maximum(Y))
        dX = (M-m)/n_bins
        dY = dX
    else
        dX = (maximum(X)-minimum(X))/n_bins
        dY = (maximum(Y)-minimum(Y))/n_bins
    end
    # bins if a semi spacial map is needed
    if angleX
        dX = 2π/n_bins
    end
    if angleY
        dY = 2π/n_bins
    end
    bin_values = zeros(n_bins,n_bins)
    bin_number = zeros(n_bins,n_bins)

    V = zeros(n_bins,n_bins,2)
    D = zeros(n_bins,n_bins,2)
    for i in 1:min(size(X,1),size(Y,1))
        # choose which are impossible
        if norm([X[i],Y[i]])<exclude
            continue
        end
        # indexing functions
        index_X = Int(ceil((X[i]-mod(X[i],dX))/dX)).+Int(n_bins/2)
        index_Y = Int(ceil((Y[i]-mod(Y[i],dY))/dY)).+Int(n_bins/2)

        if index_X > 0 && index_X <= n_bins && index_Y > 0 && index_Y <= n_bins
            # update the cound and Z number
            bin_values[index_X,index_Y] += Z[i]
            bin_number[index_X,index_Y] += 1
            # if a vector Z is used
            if vectors != nothing
                V[index_X,index_Y,:] += vectors[i,:]
                D[index_X,index_Y,:] += [X[i],Y[i]]
            end
        end
    end
    # mean of Zs
    R = bin_values./bin_number
    # fill missings
    for i in 1:size(R,1)
        for j in 1:size(R,2)
            if isnan(R[i,j])
                R[i,j] = fill_value
            end
        end
    end
    if square
        display(heatmap(linspace(m,M,n_bins),linspace(m,M,n_bins),R,aspect_ratio=:equal))
        if vectors != nothing
            return bin_values,bin_number,R,m,M,V,D
        end
        return bin_values,bin_number,R,m,M
    else
        display(heatmap(linspace(minimum(X),maximum(X),n_bins),linspace(minimum(Y),maximum(Y),n_bins),R,aspect_ratio=:equal))
        if vectors != nothing
            return bin_values,bin_number,R,V,D
        end
        return bin_values,bin_number,R
    end

end

function NearestNeighborsRelativePositions(X,T,k=1)
    x = []
    y = []
    neighbour_value = []
    # 2 for forward velocity information
    for t in 2:T
        P = X[t,:,1:2]
        data = transpose(P)
        tree = KDTree(Array(data))
        for b in 1:size(P,1)
            idxs, dists = knn(tree, P[b,1:2], k+1)
            if idxs[1] > size(X,2)
                # outside bounds for some reason
                continue
            end
            if P[idxs[1],1]-P[b,1] == 0 && P[idxs[1],2]-P[b,2] == 0
                # a null case
                continue
            end
            if P[b,1:2] != [NaN,NaN] && idxs[1] != b && P[idxs[1],1] != [NaN,NaN]
                # track a statistic to plot
                # TODO write take a lambda for this?

                # push the neighbour and its relative position
                push!(neighbour_value,a)
                # relative position
                pos = [P[idxs[1],1]-P[b,1],P[idxs[1],2]-P[b,2]]
                u = X[t,b,1:2] .- X[t-1,b,1:2]
                u = u./norm(u)
                # angle of focus velocity (heading)
                angle = atan(u[2],u[1])
                # rotation matrix to focus particles x-axis
                R = [cos(-angle) -sin(-angle); sin(-angle) cos(-angle)]
                pos = R*pos
                push!(x, pos[1])
                push!(y, pos[2])
            end
        end
    end

    NN = float.(cat(x,y,dims=2))
    NN = cat(NN,neighbour_value,dims=2)
    return NN
end

# same as python's meshgrid
# meshgrid(x,y) = (repeat(x',length(y),1),repeat(y,1,length(x)))
#
# function Grid2D(points; n_boxes=100, bounds = zeros(4))
#     """
#         For a set of points (points) construct a 2D grid covering
#         the entire area, by default, or passed bounds (bounds) with
#         (n_boxes)
#     """
#     if bounds == zeros(4)
#         min_x = minimum(points[:,1])
#         max_x = maximum(points[:,1])
#         min_y = minimum(points[:,2])
#         max_y = maximum(points[:,2])
#     else
#         min_x,max_x,min_y,max_y = bounds[1],bounds[2],bounds[3],bounds[4]
#     end
#
#     # box length
#     dx = (max_x-min_x)/n_boxes
#     dy = (max_y-min_y)/n_boxes
#
#     x = collect(min_x:dx:(max_x-dx))
#     y = collect(min_y:dy:(max_y-dy))
#
#     boxes = zeros((n_boxes^2,2,2))
#
#     # create list of boxes
#     k = 1
#     for i in x
#         for j in x
#             boxes[k,:,:] = [i i+dx ; j j+dy]
#             k += 1
#         end
#     end
#
#     # return the boxes and their x,y coordinate for the "bottom left" point
#     return boxes,x,y
# end
#
# function InBox(B, point)
#     """
#     Checks if point is in box B (Dx2 vector)
#
#     A box is a set of upper and lower bounds in D dimensional space
#     something like [l_1,u_1]x[l_2,u_2]x...x[l_D,u_D]
#
#     This is then a Dx2 vector of bounds
#     """
#     if size(B,1) != size(point,1)
#         "Need same dimension got"
#         return false
#     else
#         for i in collect(1:size(B,1))
#             if point[i] < B[i,1] || point[i] > B[i,2]
#                 return false
#             end
#         end
#         return true
#     end
# end
#
# function PointsInBox(B, X)
#     """
#     Wraps InBox to determine which of the nx2 matrix of
#     points X is in box B
#
#     A box is a set of upper and lower bounds in D dimensional space
#     something like [l_1,u_1]x[l_2,u_2]x...x[l_D,u_D]
#
#     This is then a Dx2 vector of bounds
#     """
#     In = []
#     for i in collect(1:size(X,1))
#         if InBox(B,X[i,:])
#             push!(In,i)
#         end
#     end
#     return In
# end
#
# function AssignBoxValues(points,boxes,values)
#     """
#         Takes a set of point, boxes, and associated values. Then for
#         each box finds the points in it and assigns the mean of the
#         associated values.
#
#         points  Nx2
#         boxes   Bx4
#         values  Nx1
#     """
#     mean_box_values = zeros(size(boxes,1))
#     @showprogress for i in 1:size(boxes,1)
#         ind = PointsInBox(boxes[i,:,:],points[:,1:2])
#         # ignore for no points in this box
#         if isempty(points[ind,:])
#             continue
#         else
#             mean_box_values[i] = mean(values[ind])
#         end
#     end
#     return mean_box_values
# end
#
# function AssignBoxCounts(points,boxes)
#     """
#         Takes a set of point, boxes. Assigns each box the
#         number of points in it.
#
#         points  Nx2
#         boxes   Bx4
#     """
#     mean_box_values = zeros(size(boxes,1))
#     @showprogress for i in 1:size(boxes,1)
#         ind = PointsInBox(boxes[i,:,:],points[:,1:2])
#         # ignore for no points in this box
#         if isempty(points[ind,:])
#             continue
#         else
#             mean_box_values[i] = length(points[ind,:])
#         end
#     end
#     return mean_box_values
# end
#
# function AssignBoxVectors(points,boxes,vectors)
#     """
#         Takes a set of point, boxes, and associated vectors. Then for
#         each box finds the points in it and assigns the mean of the
#         associated vectors (i.e a vector of the mean of each dimension).
#
#         points  Nx2
#         boxes   Bx4
#         values  NxD
#     """
#     mean_box_vectors = zeros(size(boxes,1),size(vectors,2))
#     @showprogress for i in 1:size(boxes,1)
#         ind = PointsInBox(boxes[i,:,:],points[:,1:2])
#         # ignore for no points in this box
#         if isempty(points[ind,:])
#             continue
#         else
#             mean_box_vectors[i,:] = mean(vectors[ind,:],dims=1)
#         end
#     end
#     return mean_box_vectors
# end
#
# function BoxCentres(Boxes)
#     """
#         Returns the centre of each 2D box
#     """
#     C = zeros(size(Boxes,1),2)
#     for i in 1:size(Boxes,1)
#         C[i,:] = [abs(Boxes[i,1,1]-Boxes[i,1,2]), abs(Boxes[i,2,1]-Boxes[i,2,2])]
#         C[i,:] = C[i,:]./2 .+ [Boxes[i,1,1],Boxes[i,2,1]]
#     end
#     return C
# end
#
# # scaling is useful to plot over the heatmaps with quiver plots for example.
# function Scaling(coords,lim)
#     """
#         Min max scaling to the VoronoiDelaunay range requirement
#     """
#     Rescale = zeros(size(coords))
#     for d in 1:2
#         min = minimum(coords[:,d])
#         max = maximum(coords[:,d])
#         a = minimum(lim[d,:])
#         b = maximum(lim[d,:])
#         Rescale[:,d] = [ ((coords[i,d] - min)/(max-min))*(b-a)+a for i in 1:size(coords,1)]
#     end
#     return Rescale
# end
#
# function HeatMapLimits(Map)
#     lims = zeros(2)
#     for i in 1:size(Map,1)
#         if isnan(Map[i,1])
#             lims[1] = i
#             break
#         end
#     end
#     for j in 1:size(Map,2)
#         if isnan(Map[1,j])
#             lims[2] = j
#             break
#         end
#     end
#     return lims
# end
#
# function HeatMap(points,n_boxes,values=ones(size(points,1)); bounds=[-1000,1000,-1000,1000],interpolation="None",step=2,counts=false)
#     """
#         Wraps around the whole module to produce a heatmap which
#         is displayed and returned as a matrix.
#
#         points          Nx2 coordinates
#         n_boxes         number of boxes to approximate with
#         values          Nx1 the desired value associated with each coordinates
#         bounds          limits on which points to consider
#         interpolation   whether to interpolate with python's scipy.interpolate.griddata
#                         can do linear,cubic, or nearest (see the scipy page)
#         step            the interpolation step
#         counts          if counts is true only the number of points in each
#                         box is assigned (position density)
#     """
#     boxes,x,y = Grid2D(points,n_boxes=n_boxes,bounds=bounds)
#
#     if (counts)
#         mean_box_values = AssignBoxCounts(points,boxes)
#     else
#         mean_box_values = AssignBoxValues(points,boxes,values)
#     end
#
#     map = zeros(n_boxes,n_boxes)
#     points = zeros(size(mean_box_values,1),2)
#     k = 1
#     for i in 1:size(x,1)
#         for j in 1:size(y,1)
#             map[i,j] = mean_box_values[k]
#             points[k,:] = [x[i],y[j]]
#             k += 1
#         end
#     end
#     if (interpolation == "linear")
#         xi = collect(bounds[1]:step:bounds[2])
#         yi = collect(bounds[3]:step:bounds[4])
#         xi,yi = meshgrid(xi,yi)
#         G = si.griddata(points,mean_box_values,(xi,yi),method="linear")
#         display(heatmap(G))
#         return G,boxes
#     end
#     if (interpolation == "cubic")
#         xi = collect(bounds[1]:step:bounds[2])
#         yi = collect(bounds[3]:step:bounds[4])
#         xi,yi = meshgrid(xi,yi)
#         G = si.griddata(points,mean_box_values,(xi,yi),method="cubic")
#         display(heatmap(G))
#         return G,boxes
#     end
#     if (interpolation == "nearest")
#         xi = collect(bounds[1]:step:bounds[2])
#         yi = collect(bounds[3]:step:bounds[4])
#         xi,yi = meshgrid(xi,yi)
#         G = si.griddata(points,mean_box_values,(xi,yi),method="nearest")
#         display(heatmap(G))
#         return G,boxes
#     end
#
#     display(heatmap(map))
#     return map,boxes
# end
#
# function HeatMapQuiver(HeatMap,points,quiver_boxes,vectors=ones(size(points,1),2);color="white",bounds=[-1000,1000,-1000,1000])
#     """
#         Adds a quiver plot of the passed vectors
#
#         It is often usefull to compute a vector field resultant
#         from a smaller number of boxes
#
#         HeatMap         A previously computed heatmap
#         points          data points
#         quiver_boxes    the number of boxes to compute the quiver on
#         vectors         the vectors to quiver plot
#         color           vector colours
#         bounds          the quiver bounds
#     """
#     boxes,x,y = Grid2D(points,n_boxes=quiver_boxes,bounds=bounds)
#     lims = HeatMapLimits(HeatMap)
#     V = AssignBoxVectors(points,boxes,vectors)
#     C = BoxCentres(boxes)
#     C = Scaling(C,[0 lims[1];0 lims[2]])
#     indx = V[:,1] .> 0
#     indy = V[:,2] .> 0
#     v = V[indx.+indy .> 0,:]
#     c = C[indx.+indy .> 0,:]
#     heatmap(M)
#     quiver!(C[:,1],C[:,2], quiver=(V[:,1],V[:,2]), color=color)
#     display(plot!(xlims=(0,lims[1]),ylims=(0,lims[2])))
# end
#
# function Ellipse(n,a=1,b=1,max=2π)
#     x = [a*cos(t) for t in 0:(2π/n):2π]
#     y = [b*sin(t) for t in 0:(2π/n):2π]
#     return cat(x,y,dims=2)
# end
#
# function PlotFocus(HeatMap,shape,color="black")
#     lims = HeatMapLimits(HeatMap)
#     scatter!([lims[1]/2],[lims[2]/2],markersize=1.0,c=color,shape=shape,label="")
# end
#
# rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
#
# function PlotBoxes(boxes)
#     plot()
#     for i in 1:size(boxes,1)
#         w = boxes[i,1,2] - boxes[i,1,1]
#         h = boxes[i,2,2] - boxes[i,2,1]
#         p = plot!(rectangle(w,h,boxes[i,1,1],boxes[i,2,1]));
#     end
#     plot!()
# end

end # module PositionalHatMaps
