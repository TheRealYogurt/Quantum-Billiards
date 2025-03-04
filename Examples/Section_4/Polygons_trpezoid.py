import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt


def trapezoid(a,b,h,):
    return pb.Polygon([[-a/2, 0], [-b/2, h], [b/2, h], [a/2, 0]])


a = 3.2
b = 1.4
h = 1.5

shape = trapezoid(a,b,h)
shape.plot()
#plt.show()



model = pb.Model(graphene.monolayer(),trapezoid(a,b,h))
model.plot()
plt.show()
