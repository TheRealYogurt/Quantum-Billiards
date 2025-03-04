import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt


def circle(radius):
    def contains(x, y, z):
        return np.sqrt(x**2 + y**2) < radius
    return pb.FreeformShape(contains, width=[2*radius, 2*radius])

radius = 2.5

shape = circle(radius)

shape.plot()

model = pb.Model(
    graphene.monolayer(),
    circle(radius)
)
model.plot()

plt.show()