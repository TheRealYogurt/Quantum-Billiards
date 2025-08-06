import pybinding as pb, matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from scipy.interpolate import griddata


def monolayer_1T_WTe2_basic_basic():
    # lattice vectors
    a = 3.477; b = 6.249; a1=[a, 0]; a2=[0, b] 

    #nsite energies d -> W, p -> Te
    mu_d = 0.74
    mu_p = -1.75 
    
    # Hopping energies in eV: 
    t_px = 1.13; t_dx = -0.41; t_pAB = 0.40; t_dAB = 0.51 
    t_0AB = 0.39; t_0ABx = 0.29; t_0x = 0.14; t_py = 0.13   


    lat = pb.Lattice( a1, a2) # Rectangular 2D lattice

    lat.add_sublattices(
        # Main cell: 
        ('W1', [ -0.25 * a, 0.32* b], mu_d), # Sublattice A 
        ('Te1',[-0.25 * a, -0.07 * b], mu_p), # Sublattice A
        ('W2', [ 0.25 * a, -0.32 * b], mu_d), # Sublattice B 
        ('Te2',[ 0.25 * a, 0.07 * b], mu_p) # Sublattice B 
    )
    

    lat.add_hoppings(
        # Main Cell Hoppings: Between atomms in the cell
        ([0, 0], 'Te1', 'W1', 0), 
        ([0, 0], 'Te2', 'W1', t_0AB), 

        ([0, 0], 'Te2', 'Te1', t_pAB),
        ([0, 0], 'W2', 'Te1', -t_0AB), 

        ([0, 0], 'Te2', 'W2', 0), 

        # Between Cells Hoppings: From inside the cell to the right (outside the cell)   
        ([1, 0], 'W1', 'W1', t_dx),
        ([1, 0], 'W2', 'W2', t_dx), 

        ([1, 0], 'Te2', 'Te2', t_px), 
        ([1, 0], 'Te1', 'Te1', t_px),

        ([1, 0], 'Te2', 'Te1', t_pAB),
        
        ([1, 0], 'Te2', 'W1', -t_0AB),
        
        ([1, 0], 'W2', 'Te1', t_0AB),

        ([1, 0], 'W1', 'Te2', t_0ABx),  
        ([1, 0], 'W1', 'Te1', t_0x, ),  

        ([1, 0], 'Te1', 'W2', -t_0ABx), 
        
        ([1, 0], 'W2', 'Te2', t_0x), 
        
        # Between Cells Hoppings: From inside the cell to the bottom (outside the cell)
        ([0, -1], 'Te1', 'W1', 0),
        
        ([0, -1], 'W2', 'W1', t_dAB),
        ([1, -1], 'W2', 'W1', t_dAB),
        
        ([0, -1], 'Te2', 'Te2', t_py),

        # Between Cells Hoppings: From inside the cell to the top (outside the cell)
        ([0, 1], 'Te2', 'W2', 0), 
        
        ([0, 1], 'Te1', 'Te1', t_py), 
                
        # Between Cells Hoppings: From inside the cell to the left (outside the cell)
        ([-1, 0], 'W2', 'Te2', -t_0x), 

        # Between Cells Hoppings: From outside the cell to the inside(from the left)
        ([-1, 0], 'W1', 'Te1', -t_0x), 

        # Between Cells Hoppings: Next cell over hoppoings(tothe left)
        ([-2, 0], 'W1', 'Te2', -t_0ABx), 
        ([-2, 0], 'Te1', 'W2', t_0ABx), 
    )

    return lat

#plt.figure(); lattice = monolayer_1T_WTe2_basic_basic(); lattice.plot() # plot he actuale lattice with the hoppings


scale = 8

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

k = 120 
model = pb.Model(monolayer_1T_WTe2_basic_basic(), shape2)
solver = pb.solver.arpack(model, k=k, sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 

eigenvalues = solver.eigenvalues # Eigen values - energies 
eigenvectors = solver.eigenvectors # Eigen vector - eigen wave functions 
positions = solver.system.positions # real space positions of each atom - i think 

r = np.column_stack(positions); steps = np.size(r[:,0]); points_2d = r[:, :2]
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
plt.scatter(r[:, 0], r[:, 1], c=np.abs(psi)**2, cmap="viridis", s=10)
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
