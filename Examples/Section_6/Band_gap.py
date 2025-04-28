import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt


def mass_term(delta):
    """Break sublattice symmetry with opposite A and B onsite energy"""
    @pb.onsite_energy_modifier
    def potential(energy, sub_id):
        energy[sub_id == 'A'] += delta
        energy[sub_id == 'B'] -= delta
        return energy
    return potential



model = pb.Model(
    graphene.monolayer(), pb.rectangle(1.2), pb.translational_symmetry(a1=True, a2=False), mass_term(delta=2.5))  # eV



plt.figure()
solver = pb.solver.lapack(model)
a = graphene.a_cc * sqrt(3)
bands = solver.calc_bands(-pi/a, pi/a)
bands.plot()

plt.show()
