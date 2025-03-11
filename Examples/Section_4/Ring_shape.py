import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt


def ring(inner_radius,outer_radius):
    def contain(x,y,z):
        r = np.sqrt(x**2+y**2)
        return np.logical_and(inner_radius<r, r<outer_radius)
    return pb.FreeformShape(contain,width = [2*outer_radius, 2*outer_radius])

shape = ring(inner_radius = 2, outer_radius=3)
#shape.plot()


model = pb.Model(graphene.monolayer(), ring(inner_radius = 2, outer_radius=3))
model.plot()
model.shape.plot()





plt.show()