
polys = [Point([-72, -36]), Point([-71.5, -38]), Point([-73, -40]), Point([-73.5, -41]), Point([-73.2, -38]), Point([-72, -39]), Point([-73.1, -40])]
a = [[shape['ID_3'][ii] for ii in range(len(blin)) if shape.geometry[ii].contains(poly)] for poly in polys]
