import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt





model = pb.Model(graphene.monolayer_4atom(), pb.rectangle(x=2, y=2), pb.translational_symmetry(a1=1.2, a2=False)) # will compine since the period length is smaller than the shape 
model.plot()

plt.figure()
model = pb.Model(graphene.monolayer_4atom(), pb.rectangle(x=2, y=2), pb.translational_symmetry(a1=2.1, a2=False)) # will compine since the period length is smaller than the shape 
model.plot()




def ring(inner_radius, outer_radius):
    def contains(x, y, z):
        r = np.sqrt(x**2 + y**2)
        return np.logical_and(inner_radius < r, r < outer_radius)
    return pb.FreeformShape(contains, width=[2*outer_radius, 2*outer_radius])


model = pb.Model(graphene.monolayer_4atom(), ring(inner_radius=1.4, outer_radius=2), pb.translational_symmetry(a1=3.8, a2=False)) #forms the audi logo because the periodic translations is smaller than the same itself  
plt.figure(figsize=[8, 3])
model.plot()

plt.figure()
solver = pb.solver.arpack(model, k=10) # only the 10 lowest states
a = 3.8  # [nm] unit cell length
bands = solver.calc_bands(-pi/a, pi/a)
bands.plot(point_labels=['$-\pi / a$', '$\pi / a$'])

plt.show()
