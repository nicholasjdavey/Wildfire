import geopandas as gp
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpp
import matplotlib.collections as mpc
import matplotlib.cm as clrmp
import matplotlib.backends.backend_pdf as pdf

import pyproj
import json
from shapely.geometry import Point, mapping
from functools import partial
from shapely.ops import transform

shape = gp.GeoDataFrame.from_file("CHL_adm2.shp")
selectedProvinces = [8, 9, 15, 16, 17, 18, 25, 26, 27, 28, 29, 30, 37, 38, 39, 40]
stations = [[-73.8, -42], [-71, -36], [-73, -39], [-73.5, -40.5], [-72.6, -36.1], [-71.3, -38.4]]
shape['Relevant'] = [(True if ii in selectedProvinces else False) for ii in range(1, len(shape) + 1)]
relevant = shape[shape['Relevant']]

first = shape.geometry[selectedProvinces[0]]

xMin = first.bounds[0]
yMin = first.bounds[1]
xMax = first.bounds[2]
yMax = first.bounds[3]

for provinceID in selectedProvinces:
    province = shape.geometry[provinceID - 1]
    xMinP = province.bounds[0]
    yMinP = province.bounds[1]
    xMaxP = province.bounds[2]
    yMaxP = province.bounds[3]

    xMax = xMaxP if xMaxP > xMax else xMax
    yMax = yMaxP if yMaxP > yMax else yMax
    xMin = xMinP if xMinP < xMin else xMin
    yMin = yMinP if yMinP < yMin else yMin

xSpan = xMax - xMin
ySpan = yMax - yMin

outputGraphs = pdf.PdfPages("test.pdf")

fig = plt.figure(figsize=(5,5*(yMax - yMin)/(xMax - xMin)))
ax = fig.add_subplot(111)
ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)
ax.set_aspect('equal')

local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"

wgs84_to_aeqd = partial(
    pyproj.transform,
    pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
    pyproj.Proj(local_azimuthal_projection),
)

aeqd_to_wgs84 = partial(
    pyproj.transform,
    pyproj.Proj(local_azimuthal_projection),
    pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
)

radii = []

for base in stations:
    point = Point(base[0], base[1])

    point_transformed = transform(wgs84_to_aeqd, point)

    buff = point_transformed.buffer(120000)
    buffer_wgs84 = transform(aeqd_to_wgs84, buff)
    radii.append(buffer_wgs84)

#    x, y = buffer_wgs84.exterior.xy
#    ax.plot(x, y, color='black')

patches = []

cmap = clrmp.get_cmap('Oranges')

#shape.loc[[provinceID-1], 'geometry'].plot(ax = ax, edgecolor='black', color=cmap(numpy.random.rand()))
for regionID in relevant:
    newRegions.loc[[regionID], 'geometry'].plot(ax = ax, edgecolor='black', color=cmap(numpy.random.rand()))

basePolys = []
# Annotate with bases
for base in stations:
    polygon = mpp.Polygon(
            [(base[0] - 1.05*xSpan/40, base[1] - 0.95*xSpan/40),
             (base[0] - 0.95*xSpan/40, base[1] - 1.05*xSpan/40),
             (base[0] + 1.05*xSpan/40, base[1] + 0.95*xSpan/40),
             (base[0] + 0.95*xSpan/40, base[1] + 1.05*xSpan/40)],
            closed=True)
    basePolys.append(polygon)

p = mpc.PatchCollection(basePolys)
p.set_array(numpy.ones(len(stations)))
ax.add_collection(p)

for base in basePolys:
    ax.add_patch(mpp.Polygon(base.get_xy(),
                 closed=True,
                 ec='b',
                 lw=1,
                 fill='b'))

fig.canvas.draw()

outputGraphs.savefig(fig)
outputGraphs.close()
