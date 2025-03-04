import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt


def rectangle(width, height):
    x0 = width / 2
    y0 = height / 2
    return pb.Polygon([[x0, y0], [x0, -y0], [-2*x0, -y0], [-2*x0, y0]])

shape = rectangle(1.6, 1.2)
shape.plot()
#plt.show()



model = pb.Model(graphene.monolayer(),rectangle(1.6, 1.2))
model.plot()
plt.show()


