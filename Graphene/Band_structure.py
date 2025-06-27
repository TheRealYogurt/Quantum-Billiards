import pybinding as pb, matplotlib.pyplot as plt, numpy as np, pandas as pd, multiprocessing, sys
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

k = 200 * scale; a = graphene.a
model = pb.Model(graphene.monolayer(), shape)
solver = pb.solver.arpack(model, k=k, sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 

eigenvalues = solver.eigenvalues # Eigen values - energies 
eigenvectors = solver.eigenvectors # Eigen vector - eigen wave functions 
positions = solver.system.positions # real space positions of each atom - i think 

fscale = 8
r = np.column_stack(positions); steps = int(k)
kx = np.linspace(0, 2*fscale*np.pi, steps)
ky = np.linspace(0, 2*fscale*np.pi, steps)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

mesh_step = kx[1]-kx[0]

def process_n(args):
    n, counter, lock, total = args
    psi = eigenvectors[:, n]
    M_xy = np.zeros((steps, steps), dtype=complex)
    for i in range(steps):
        for j in range(steps):
            fourier_phase = np.exp(-1j * (kx[i] * r[:, 0] + ky[j] * r[:, 1])) # This is the fourier phase 
            M_xy[i, j] = np.dot(psi, fourier_phase) * mesh_step**2 # This is the full integral 
    M_xy /= np.sqrt(len(psi))

    idx = np.unravel_index(np.abs(M_xy).argmax(), M_xy.shape)

    with lock:
        counter.value += 1
        print(f"{counter.value} out of {total}", file=sys.stderr) # counter to check the progress in the loop 

    return (kx[idx[0]], ky[idx[1]], eigenvalues[n])

if __name__ == "__main__": # initialising multiprocessing
    total = int(k)
    with multiprocessing.Manager() as manager: 
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        with multiprocessing.Pool() as pool:
            args = [(n, counter, lock, total) for n in range(total)]
            results = pool.map(process_n, args)

        max_kx_values, max_ky_values, energy_values = zip(*results)

k_matrix = np.column_stack((max_kx_values, max_ky_values, eigenvalues))
#k_matrix = k_matrix[k_matrix[:, 0] <= (6*pi / (3*a))] # filter out kx values greater than Gamma2
#k_matrix = k_matrix[k_matrix[:, 0] >= 0] # filter out negative kx values
#k_matrix = k_matrix[k_matrix[:, 1] <= (np.pi /(np.sqrt(3)*a))] # filter out ky values greater than y part of M
#k_matrix = k_matrix[k_matrix[:, 1] >= (-np.pi /(np.sqrt(3)*a))] # filter out ky values lower than y part of -M



Gamma1 = [0, 0]
#K1 = [-4*pi / (3*a), 0] #remove this to get x-axis starting from 0 
M = [pi / a, 0]
K2 = [4*pi / (3*a),0]
Gamma2 = [(6*np.pi)/(3*a), 0]
points = np.array([Gamma1, K2, M, Gamma2])
labels = ['Γ', 'K', 'M', 'Γ/K']


plt.figure()
plt.scatter(k_matrix[:, 0], k_matrix[:, 2])
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("kx - points"); plt.ylabel("Energy")

plt.scatter(points[:,0], points[:,1], c='red', marker='o')
for i, label in enumerate(labels):
    plt.text(points[i,0], points[i,1], f' {label}', fontsize=10)
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
 



