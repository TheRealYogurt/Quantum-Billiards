import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt

""" 

@pb.onsite_energy_modifier
def potential(x, y):
    return np.sin(x)**2 + np.cos(y)**2


model = pb.Model(graphene.monolayer(), pb.rectangle(12), potential) 

model.onsite_map.plot_contourf()
pb.pltutils.colorbar(label="U (eV)")

"""


def wavy2(a, b):
    @pb.onsite_energy_modifier
    def potential(energy, x, y):
        v = np.sin(a * x)**2 + np.cos(b * y)**2
        return energy + v
    return potential



model = pb.Model(graphene.monolayer(), pb.regular_polygon(num_sides=6, radius=8), wavy2(a=0.6, b=0.9))
model.onsite_map.plot_contourf()
pb.pltutils.colorbar(label="U (eV)")




plt.show()
