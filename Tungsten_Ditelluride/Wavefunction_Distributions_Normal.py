import pybinding as pb, matplotlib.pyplot as plt, numpy as np
from pybinding.repository import group6_tmd
from scipy.interpolate import griddata

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

k = 200 * scale
model = pb.Model(group6_tmd.monolayer_3band("WTe2"), shape2)
solver = pb.solver.arpack(model, k=k, sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 

eigenvalues = solver.eigenvalues # Eigen values - energies 
eigenvectors = solver.eigenvectors # Eigen vector - eigen wave functions 
positions = solver.system.positions # real space positions of each atom - i think 


raw_pos = np.column_stack(positions)
num_orbitals = eigenvectors.shape[0] // raw_pos.shape[0]
points_2d = np.repeat(raw_pos[:, :2], num_orbitals, axis=0)


x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
grid_x, grid_y = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
n = 0; psi = eigenvectors[:, n] # eigen fucntion, index 0 being the fisrt

grid_z = griddata(points_2d, np.abs(psi)**2, (grid_x, grid_y), method='linear', fill_value=0)

plt.figure(figsize=(6, 5))
plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='viridis')
plt.xlabel("x [nm]"); plt.ylabel("y [nm]")
plt.title(rf"$|\psi_{{n={n}}}(x, y)|^2$: Real-space Eigenfunction - Contourf")
plt.colorbar(label=r"$|\psi_n|^2$")
plt.axis("equal"); 
plt.tight_layout()

plt.figure(figsize=(6, 5))
plt.scatter(points_2d[:, 0], points_2d[:, 1], c=np.abs(psi)**2, cmap="viridis", s=10)
plt.xlabel("x [nm]"); plt.ylabel("y [nm]")
plt.title(rf"$|\psi_{{n={n}}}(x, y)|^2$: Real-space Eigenfunction - scatter")
plt.colorbar(label=r"$|\psi_n(x, y)|^2$")
plt.axis("equal")  # ensures correct spatial aspect ratio
plt.tight_layout()

plt.figure(figsize=(6, 5))
plt.tricontourf(points_2d[:, 0], points_2d[:, 1], np.abs(psi)**2, levels=100, cmap='viridis')
plt.xlabel('x [nm]')
plt.ylabel('y [nm]')
plt.title(r'$|\psi_{n=0}(x, y)|^2$: Real-space Eigenfunction - tricontourf')
plt.colorbar(label=r'$|\psi|^2$')
plt.axis('equal')
plt.tight_layout()


plt.show()


