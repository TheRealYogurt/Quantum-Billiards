import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt

model = pb.Model(graphene.monolayer(), pb.rectangle(1.2), pb.translational_symmetry(a1=True, a2=False))
model.plot()
model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm

plt.figure()
solver = pb.solver.lapack(model)
a = graphene.a_cc * sqrt(3)  # ribbon unit cell length
bands = solver.calc_bands(-pi/a, pi/a)
bands.plot()

plt.figure()
model = pb.Model( graphene.monolayer(), pb.rectangle(1.2), pb.translational_symmetry(a1=False, a2=True))
model.plot()
model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm


plt.figure()
model = pb.Model(graphene.monolayer_4atom(), pb.primitive(a1 = 5), pb.translational_symmetry(a1 = False))
model.plot()
model.lattice.plot_vectors(position = [-0.59, -0.6] ) 

plt.figure()
solver = pb.solver.lapack(model)
d = 3 * graphene.a_cc  # ribbon unit cell length
bands = solver.calc_bands([0, -pi/d], [0, pi/d])
bands.plot(point_labels=['$-\pi / 3 a_{cc}$', '$\pi / 3 a_{cc}$'])


plt.show()

