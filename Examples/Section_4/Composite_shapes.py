import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt

rectangle = pb.rectangle(x=6, y=1)
hexagon = pb.regular_polygon(num_sides=10, radius=1.92, angle=np.pi/6)
circle = pb.circle(radius=0.6)


shape = rectangle + hexagon - circle

model = pb.Model(graphene.monolayer(), shape)
#model.shape.plot()
#model.plot()


solver = pb.solver.arpack(model, k=20)  # only the 20 lowest eigenstates


ldos = solver.calc_spatial_ldos(energy=0, broadening=0.10)  # eV
ldos.plot(site_radius=(0.03, 0.12))
pb.pltutils.colorbar(label="LDOS")

plt.show()

