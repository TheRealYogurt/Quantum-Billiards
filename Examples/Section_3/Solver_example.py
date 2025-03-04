import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import sqrt, pi



model = pb.Model(graphene.monolayer(), pb.translational_symmetry()) #loads the graphene model #adds the infinte expansion 
solver = pb.solver.lapack(model) # apply the solver onto the graphene prelaod

eigenvals = solver.eigenvalues # calculates the eigen-values 
eigenvecs = solver.eigenvectors # calculates the eigen-vectors

#model.plot() # plot the graphene lattice

a_cc = graphene.a_cc # carbon - carbon distance

Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(K1, Gamma, M, K2)
bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])






#print(eigenvals,eigenvecs,a_cc)
plt.show()