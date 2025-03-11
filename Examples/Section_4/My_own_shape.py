import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt

# circle with a hexagon cutout  inside of a triangle

hexagon = pb.regular_polygon(num_sides=6, radius=1.92, angle=np.pi/6)
circle = pb.circle(radius=4)
triangle = pb.regular_polygon(num_sides=3 ,radius=9, angle=np.pi/3)
rectangle = pb.rectangle(x=10, y=1)

shape = triangle - circle + hexagon + rectangle
shape.plot()


plt.figure()
model = pb.Model(graphene.monolayer(),shape)
model.shape.plot()
model.plot()


plt.figure()
solver = pb.solver.arpack(model, k=20)
ldos = solver.calc_spatial_ldos(energy=0.5, broadening=0.05)  # LDOS around 0 eV
ldos.plot(site_radius=(0.03, 0.12))
pb.pltutils.colorbar(label="LDOS")

plt.show()
