import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt



width = 2.5
rectangle = pb.rectangle(x=width * 1.2, y=width * 1.2)
dot = pb.circle(radius=0.4)

model = pb.Model(graphene.monolayer_4atom(), rectangle - dot, pb.translational_symmetry(a1=width, a2=width))
plt.figure(figsize=(5, 5))
model.plot()
model.lattice.plot_vectors(position=[2, -3.5], scale=3)




