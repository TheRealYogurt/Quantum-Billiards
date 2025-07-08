import pybinding as pb, matplotlib.pyplot as plt, numpy as np
from pybinding.repository import graphene 
from math import pi, sqrt

#shape = pb.rectangle()

model = pb.Model(graphene.monolayer_4atom(), pb.translational_symmetry())

# Eigenvalues and Eigenfunctions for the system in real space 
solver = pb.solver.lapack(model)
wavefunctions = solver.eigenvectors
eigenvalues = solver.eigenvalues

# Points for graphene reciprocal space: 
a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]


# The Wavefunction calculated in reciprocal space: 
Solver_Wave = solver.calc_wavefunction(Gamma, K1, M, K2)

wavefunction_k = Solver_Wave.wavefunction # The wave function at each specific point in k-space (wavefunction_k, band_num, site_num)

k_path = Solver_Wave.bands.k_path # The k-space path calculated (kx, ky)


# The berry connection calculations: 
def berry_connection_1d(wavefunction_k, k_path, band_index):
    Nk = wavefunction_k.shape[0]
    A_n = np.zeros(Nk - 1)

    for idx in range(Nk - 1):
        psi_i = wavefunction_k[idx, band_index, :]
        psi_ip1 = wavefunction_k[idx + 1, band_index, :]

        overlap = np.vdot(psi_i, psi_ip1)
        delta_k = np.linalg.norm(k_path[idx + 1] - k_path[idx])

        A_n[idx] = np.imag(overlap) / delta_k

    return A_n

A_k = berry_connection_1d(wavefunction_k, k_path, band_index=2)

#print(sum(A_k))

