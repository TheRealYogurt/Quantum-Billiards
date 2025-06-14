import pybinding as pb, matplotlib.pyplot as plt, numpy as np, pandas as pd
from pybinding.repository import graphene
from math import pi, sqrt
from numpy.polynomial import Polynomial
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

scale = 1

###################### Circular Billiard ###################### 
circle =  pb.circle(radius=2*scale) # define a circle with radius 1nm

shape1 = circle; #plt.figure(); shape.plot()
###################### Circular Billiard ######################

###################### Africa Billiard ######################
x0 = 1.1 * scale; y0 = 0.5* scale; shifty = 1.44* scale; shiftx = -0.6* scale
circle1 =  pb.circle(radius=2* scale) # define a circle with radius 1nm
circle1 =  pb.circle(radius=2.1* scale) # define a circle with radius 1nm
circle2 = pb.circle(radius=1.31* scale, center=[-1.75* scale, 0.66* scale]) # define a circle with radius 1nm
rectangle = pb.Polygon([[x0 + shiftx, y0+ shifty], [x0+ shiftx, -y0+ shifty], [-x0+ shiftx, -y0+ shifty], [-x0+ shiftx, y0+ shifty]])

shape2 = circle1 + circle2 + rectangle; #plt.figure(); shape.plot()
###################### Africa Billiard ######################

###################### 4 Circle Billiard ######################
x = 1 * scale; y = 1 * scale ; rad = 2 * scale 
circle1 = pb.circle(radius= rad, center=[x,y]); circle2 = pb.circle(radius= rad, center=[-x,-y]) 
circle3 = pb.circle(radius= rad, center=[-x,y]); circle4 = pb.circle(radius= rad, center=[x,-y])

shape3 = circle1 + circle2 + circle3 + circle4; #plt.figure(); shape.plot()
###################### 4 Circle Billiard ######################

###################### 5 Circle Billiard ######################
x = 1 * scale; y = 1 * scale ; rad = 2 * scale 
circle1 = pb.circle(radius= rad, center=[x,y]); circle2 = pb.circle(radius= rad, center=[-x,-y]) 
circle3 = pb.circle(radius= rad, center=[-x,y]); circle4 = pb.circle(radius= rad, center=[x,-y])
circle5 = pb.circle(radius=rad*0.25, center=[-x/2, y/4])

shape4 = circle1 + circle2 + circle3 + circle4 - circle5; #plt.figure(); shape.plot()
###################### 5 Circle Billiard ######################

###################### 6 Circle Billiard ######################
x = 1 * scale; y = 1 * scale ; rad = 2 * scale 
circle1 = pb.circle(radius= rad, center=[x,y]); circle2 = pb.circle(radius= rad, center=[-x,-y]) 
circle3 = pb.circle(radius= rad, center=[-x,y]); circle4 = pb.circle(radius= rad, center=[x,-y])
circle5 = pb.circle(radius=rad*0.25, center=[-x/2, y/4]); circle6 = pb.circle(radius=rad*0.25, center=[x/4, -y/2]) 

shape5 = circle1 + circle2 + circle3 + circle4 - circle5 - circle6;  #plt.figure; shape.plot()
###################### 6 Circle Billiard ######################


model = pb.Model(graphene.monolayer(), shape2); k = 350 * scale
solver = pb.solver.arpack(model, k=int(k), sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 

eigenvalues = solver.eigenvalues # Eigen values - energies 
eigenvectors = solver.eigenvectors # Eigen vector - eigen wave functions 
positions = solver.system.positions # real space positions of each atom - i think 

fscale = 8
r = np.column_stack(positions); steps = 4*np.size(r[:,0])
kx = np.linspace(-fscale*np.pi, fscale*np.pi, steps)
ky = np.linspace(-fscale*np.pi, fscale*np.pi, steps)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

mesh_step = kx[1]-kx[0]

#"""This will produce a single plot for a specified eigensate 

n = 0; psi = eigenvectors[:, n] # eigen fucntion, index 0 being the fisrt

#calculates the fourrier distrbution i think 
M_xy = np.zeros((steps,steps),dtype=complex)

for i in range(steps):
    for j in range(steps):
        fourier_phase = np.exp(-1j * (kx[i] * r[:, 0] + ky[j] * r[:, 1]))
        M_xy[i,j] = np.dot(psi,fourier_phase) * mesh_step**2 # this should be the integral of the fourier transform
    print(i,steps)         
        
M_xy /= np.sqrt(len(psi))

# Plot magnitue of M_xy
plt.figure(figsize=(6,5))
plt.contourf(kx, ky, np.abs(M_xy.T)**2, levels=100, cmap='plasma')
plt.xlabel("$k_x$"); plt.ylabel("$k_y$"); plt.title(rf"$|M_{{n={n}}}(k_x, k_y)|^2$: Momentum Distribution")
plt.colorbar(label=r"$|M_n(k_x, k_y)|^2$") ; plt.show()
#""" 

""" This will produce a .gif for all the momentum distributions for each eigen state in a chosen shape

fig, ax = plt.subplots(figsize=(6, 5))
n_values = list(range(int(k)))  
cbar = None  
def update(frame):
    global cbar  

    ax.clear()

    n = n_values[frame]
    psi = eigenvectors[:, n]

    M_xy = np.zeros((steps, steps), dtype=complex)
    for i in range(steps):
        for j in range(steps):
            fourier_phase = np.exp(-1j * (kx[i] * r[:, 0] + ky[j] * r[:, 1]))
            M_xy[i, j] = np.dot(psi, fourier_phase) * mesh_step**2

    M_xy /= np.sqrt(len(psi))
    data = np.abs(M_xy.T)**2

    # Plot updated contour
    contour = ax.contourf(kx, ky, data, levels=100, cmap='plasma')

    # Update axis labels and title
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_title(rf"$|M_{{n={n}}}(k_x, k_y)|^2$: Momentum Distribution")

    # Remove old colorbar if it exists
    if cbar:
        cbar.remove()

    # Create new colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r"$|M_n(k_x, k_y)|^2$")

    return contour.collections + [cbar]

# Run animation
ani = FuncAnimation(fig, update, frames=len(n_values), repeat=False)
ani.save("momentum_distribution.gif", writer=PillowWriter(fps=1))

#plt.show()
#"""


