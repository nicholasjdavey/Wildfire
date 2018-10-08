import geopandas as gp
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpp
import matplotlib.collections as mpc
import matplotlib.cm as clrmp
import matplotlib.backends.backend_pdf as pdf

shape = gp.GeoDataFrame.from_file("CHL_adm3.shp")
selectedProvinces = [20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238]
stations = [[-73.8, -42], [-71, -36], [-73, -39], [-73.5, -40.5], [-72.6, -36.1], [-71.3, -38.4]]

first = shape.geometry[selectedProvinces[0]]

xMin = first.bounds[0]
yMin = first.bounds[1]
xMax = first.bounds[2]
yMax = first.bounds[3]

for provinceID in selectedProvinces:
    province = shape.geometry[provinceID - 1]
    xMinP = first.bounds[0]
    yMinP = first.bounds[1]
    xMaxP = first.bounds[2]
    yMaxP = first.bounds[3]

    xMax = xMaxP if xMaxP > xMax else xMax
    yMax = yMaxP if yMaxP > yMax else yMax
    xMin = xMinP if xMinP < xMin else xMin
    yMin = yMinP if yMinP < yMin else yMin

xMin = -76
yMin = -45
xMax = -70
yMax = -34

xSpan = xMax - xMin
ySpan = yMax - yMin

outputGraphs = pdf.PdfPages("test.pdf")

fig = plt.figure(figsize=(10,10*(yMax - yMin)/(xMax - xMin)))
ax = fig.add_subplot(111)
ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)
ax.set_aspect('equal')

patches = []

cmap = clrmp.get_cmap('Oranges')

for provinceID in selectedProvinces:
    shape.loc[[provinceID-1], 'geometry'].plot(ax = ax, edgecolor='black', color=cmap(numpy.random.rand()))

basePolys = []
# Annotate with bases
for base in stations:
    polygon = mpp.Polygon(
            [(base[0] - 1.1*xSpan/40, base[1] - 0.9*ySpan/40),
             (base[0] - 0.9*xSpan/40, base[1] - 1.1*ySpan/40),
             (base[0] + 1.1*xSpan/40, base[1] + 0.9*ySpan/40),
             (base[0] + 0.9*xSpan/40, base[1] + 1.1*ySpan/40)],
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
