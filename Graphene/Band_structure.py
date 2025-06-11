import pybinding as pb, matplotlib.pyplot as plt, numpy as np, pandas as pd
from pybinding.repository import graphene
from math import pi, sqrt
from numpy.polynomial import Polynomial
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


scale = 1; x0 = 1.1 * scale; y0 = 0.5* scale; shifty = 1.44* scale; shiftx = -0.6* scale
circle1 =  pb.circle(radius=2.1* scale) # define a circle with radius 1nm
circle2 = pb.circle(radius=1.31* scale, center=[-1.75* scale, 0.66* scale]) # define a circle with radius 1nm
rectangle = pb.Polygon([[x0 + shiftx, y0+ shifty], [x0+ shiftx, -y0+ shifty], [-x0+ shiftx, -y0+ shifty], [-x0+ shiftx, y0+ shifty]])
shape = circle1 + circle2 + rectangle; 

k = 300 * scale; a = graphene.a
model = pb.Model(graphene.monolayer(), shape)
solver = pb.solver.arpack(model, k=k, sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 

eigenvalues = solver.eigenvalues # Eigen values - energies 
eigenvectors = solver.eigenvectors # Eigen vector - eigen wave functions 
positions = solver.system.positions # real space positions of each atom - i think 

r = np.column_stack(positions); steps = int(k)
kx = np.linspace(0, 2*np.pi, steps)
ky = np.linspace(-2*np.pi, 2*np.pi, steps)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

mesh_step = kx[1]-kx[0]

max_kx_values = []; max_ky_values = []

for n in range(int(k)):
    psi = eigenvectors[:, n] # eigen fucntion, index 0 being the fisrt
    M_xy = np.zeros((steps,steps),dtype=complex)

    for i in range(steps):
        for j in range(steps):
            fourier_phase = np.exp(-1j * (kx[i] * r[:, 0] + ky[j] * r[:, 1]))
            M_xy[i,j] = np.dot(psi,fourier_phase) * mesh_step**2 # this should be the integral of the fourier transform 
            
    M_xy /= np.sqrt(len(psi))

    idx = np.unravel_index(np.abs(M_xy).argmax(), M_xy.shape)
    
    max_kx_values.append(kx[idx[0]]); max_ky_values.append(ky[idx[1]])
    print(n) #iterations counter to check progess in the loop



plt.figure()
plt.scatter(max_kx_values, max_ky_values)
plt.xlabel("Kx"); plt.ylabel("Y-axis")

""" 
a_cc = 1
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

points = np.array([K1, Gamma, M, K2])
labels = ['K1', 'Γ', 'M', 'K2']
"""
""" 
a_cc = 1
Gamma = [0, 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

points = np.array([Gamma, M, K2])
labels = ['Γ', 'M', 'K2']



plt.scatter(points[:,0], points[:,1], c='red', marker='o')
for i, label in enumerate(labels):
    plt.text(points[i,0], points[i,1], f' {label}', fontsize=12)

"""

plt.show()







""" 
#------------- PLOTS THE BAND STRUCTURE ----------------------#
a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(K1, Gamma, M, K2); plt.figure()
bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
"""
 



