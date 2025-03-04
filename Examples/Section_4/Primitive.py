import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt



a1 = 5
a2 = 5

model = pb.Model(graphene.bilayer(),pb.primitive(a1,a2))
model.plot()
model.lattice.plot_vectors(position=[0.6, -0.25])
plt.show()