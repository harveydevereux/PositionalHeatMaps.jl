## PositionalHeatMaps.jl
#### Heatmaps where heat values are plotted at the x,y 'positions' of data points.

#### Examples

Below see a PositionalHeatMap of the effect of the nearest neighbours position on the focus beetle's speed.

I.e. each point from the center represent the relative position of the nearest neighbour within the reference frame
of a focus beetle in the centre, at each point the colour is the focus beetles speed when the nearest neighbour is in that
position. This is averaged over a 3 minute video of 200 beetles by taking all pairs.

![nearest neighbour position effect on speed](https://raw.githubusercontent.com/harveydevereux/PositionalHeatMaps.jl/master/resources/NN-focus-speed.png)
