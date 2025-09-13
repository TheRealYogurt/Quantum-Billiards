import pybinding as pb, matplotlib.pyplot as plt, numpy as np
from scipy.interpolate import griddata


# With SOC:
def monolayer_1T_WTe2_SOC():
    # lattice vectors
    a = 3.477; b = 6.249; a1=[a, 0]; a2=[0, b] 

    #nsite energies d -> W, p -> Te
    mu_d = 0.74; mu_d_spinor = mu_d * np.identity(2)
    mu_p = -1.75; mu_p_spinor = mu_p * np.identity(2)
    
    # Hopping energies in eV: 
    t_px = 1.13; t_dx = -0.41; t_pAB = 0.40; t_dAB = 0.51 
    t_0AB = 0.39; t_0ABx = 0.29; t_0x = 0.14; t_py = 0.13   

    # Spin Orbit Energies in eV:
    lambda_0AB_y = 0.011; lambda_0_y = 0.051; lambda_0_z = 0.012
    lambda_0_y_p = 0.050; lambda_0_z_p = 0.012; lambda_px_y = -0.040 
    lambda_pxz = -0.010; lambda_dx_y = -0.031; lambda_dx_z = -0.008 

    # Pauli Matricies: 
    complex_matrix = np.full((2, 2), 1j)

    sigma_0 = np.array([[1, 0], [0, 1]]); sigma_0_i = sigma_0
    sigma_x = np.array([[0, 1], [1, 0]]); sigma_x_i = np.multiply(sigma_x, complex_matrix)
    sigma_y = np.array([[0, -1j], [1j, 0]]); sigma_y_i = np.multiply(sigma_y, complex_matrix)
    sigma_z = np.array([[1, 0], [0, -1]]); sigma_z_i = np.multiply(sigma_z, complex_matrix)
  
    lat = pb.Lattice( a1, a2) # Rectangular 2D lattice

    lat.add_sublattices(
        # Main cell: 
        ('W1', [ -0.25 * a, 0.32* b], mu_d_spinor), # Sublattice A 
        ('Te1',[-0.25 * a, -0.07 * b], mu_p_spinor), # Sublattice A
        ('W2', [ 0.25 * a, -0.32 * b], mu_d_spinor), # Sublattice B 
        ('Te2',[ 0.25 * a, 0.07 * b], mu_p_spinor) # Sublattice B 
    )
    

    # Function to calculate the hopping parameter
    def hopping_parameter(t, lambda_x, lambda_y, lambda_z): 
        parameter =  np.add(np.add(np.multiply(t,sigma_0_i), np.multiply(sigma_x_i, lambda_x)), 
                            np.add(np.multiply(sigma_y_i, lambda_y), np.multiply(sigma_z_i, lambda_z)))
        return parameter

    lat.add_hoppings(
        # Main Cell Hoppings: Between atomms in the cell
        ([0, 0], 'Te1', 'W1', hopping_parameter(t = 0, lambda_x = 0, lambda_y = -lambda_0_y, lambda_z = -lambda_0_z)), 
        ([0, 0], 'Te2', 'W1', hopping_parameter( t = t_0AB, lambda_x = 0, lambda_y = -lambda_0AB_y, lambda_z = 0)), 

        ([0, 0], 'Te2', 'Te1', hopping_parameter(t = t_pAB, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        ([0, 0], 'W2', 'Te1', hopping_parameter(t = -t_0AB, lambda_x = 0, lambda_y = -lambda_0AB_y, lambda_z = 0)), 

        ([0, 0], 'Te2', 'W2', hopping_parameter(t = 0, lambda_x = 0, lambda_y = lambda_0_y, lambda_z = lambda_0_z)), 

        # Between Cells Hoppings: From inside the cell to the right (outside the cell)   
        ([1, 0], 'W1', 'W1', hopping_parameter(t = t_dx, lambda_x = 0, lambda_y = -lambda_dx_y, lambda_z = -lambda_dx_z)),
        ([1, 0], 'W2', 'W2', hopping_parameter(t = t_dx, lambda_x = 0, lambda_y = lambda_dx_y, lambda_z = lambda_dx_z)), 

        ([1, 0], 'Te2', 'Te2', hopping_parameter(t = t_px, lambda_x = 0, lambda_y = lambda_px_y, lambda_z = lambda_pxz)), 
        ([1, 0], 'Te1', 'Te1', hopping_parameter(t = t_px, lambda_x = 0, lambda_y = -lambda_px_y, lambda_z = -lambda_pxz )),

        ([1, 0], 'Te2', 'Te1', hopping_parameter(t = t_pAB, lambda_x = 0, lambda_y = 0, lambda_z = 0 )),
        
        ([1, 0], 'Te2', 'W1', hopping_parameter(t = -t_0AB, lambda_x = 0, lambda_y = -lambda_0AB_y, lambda_z = 0)),
        
        ([1, 0], 'W2', 'Te1', hopping_parameter(t = t_0AB , lambda_x = 0, lambda_y = -lambda_0AB_y, lambda_z = 0)),

        ([1, 0], 'W1', 'Te2', hopping_parameter(t = t_0ABx, lambda_x = 0, lambda_y = 0, lambda_z = 0)),  
        ([1, 0], 'W1', 'Te1', hopping_parameter(t = t_0x, lambda_x = 0, lambda_y = 0, lambda_z = 0)),  

        ([1, 0], 'Te1', 'W2', hopping_parameter(t = -t_0ABx, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        
        ([1, 0], 'W2', 'Te2', hopping_parameter(t = t_0x, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        
        # Between Cells Hoppings: From inside the cell to the bottom (outside the cell)
        ([0, -1], 'Te1', 'W1', hopping_parameter(t = 0, lambda_x = 0, lambda_y = -lambda_0_y_p, lambda_z = -lambda_0_z_p)),
        
        ([0, -1], 'W2', 'W1', hopping_parameter(t = t_dAB, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        ([1, -1], 'W2', 'W1', hopping_parameter(t = t_dAB, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        
        ([0, -1], 'Te2', 'Te2', hopping_parameter(t = t_py, lambda_x = 0, lambda_y = 0, lambda_z = 0)),

        # Between Cells Hoppings: From inside the cell to the top (outside the cell)
        ([0, 1], 'Te2', 'W2', hopping_parameter(t = 0, lambda_x = 0, lambda_y = lambda_0_y_p, lambda_z = lambda_0_z_p)), 
        
        ([0, 1], 'Te1', 'Te1', hopping_parameter(t = t_py, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
                
        # Between Cells Hoppings: From inside the cell to the left (outside the cell)
        ([-1, 0], 'W2', 'Te2', hopping_parameter(t = -t_0x, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 

        # Between Cells Hoppings: From outside the cell to the inside(from the right)
        ([-1, 0], 'W1', 'Te1', hopping_parameter(t = -t_0x, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 

        # Between Cells Hoppings: Next cell over hoppoings(tothe left)
        ([-2, 0], 'W1', 'Te2', hopping_parameter(t = -t_0ABx, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        ([-2, 0], 'Te1', 'W2', hopping_parameter(t = t_0ABx, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
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

k = 280 
model = pb.Model(monolayer_1T_WTe2_SOC(), shape2)
solver = pb.solver.arpack(model, k=k, sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 

eigenvalues = solver.eigenvalues # Eigen values - energies 
eigenvectors = solver.eigenvectors # Eigen vector - eigen wave functions 
positions = solver.system.positions # real space positions of each atom - i think 

r = np.column_stack(positions)
n = 17; psi = eigenvectors[:, n] # eigen fucntion, index 0 being the fisrt

positions = np.array(positions).T
x = positions[:, 0]
y = positions[:, 1]
points_2d = np.column_stack((x, y))

psi_spinor = psi.reshape((len(positions), 2))

# Total probability density:
psi_abs_squared = np.sum(np.abs(psi_spinor)**2, axis=1)

# Grid for interpolation
grid_x, grid_y = np.mgrid[min(x):max(x):300j, min(y):max(y):300j]
grid_z = griddata(points_2d, psi_abs_squared, (grid_x, grid_y), method='linear', fill_value=0)


plt.figure(figsize=(6, 5))
plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='viridis')
plt.xlabel("x [nm]")
plt.ylabel("y [nm]")
plt.title(rf"$|\psi_{{n={n}}}(x, y)|^2$: Contourf")
plt.colorbar(label=r"$|\psi_n|^2$")
plt.axis("equal")
plt.tight_layout()

plt.figure(figsize=(6, 5))
plt.scatter(x, y, c=psi_abs_squared, cmap="viridis", s=10)
plt.xlabel("x [nm]")
plt.ylabel("y [nm]")
plt.title(rf"$|\psi_{{n={n}}}(x, y)|^2$: Scatter")
plt.colorbar(label=r"$|\psi_n(x, y)|^2$")
plt.axis("equal")
plt.tight_layout()

plt.figure(figsize=(6, 5))
plt.tricontourf(points_2d[:, 0], points_2d[:, 1], psi_abs_squared, levels=100, cmap='viridis')
plt.xlabel('x [nm]')
plt.ylabel('y [nm]')
plt.title(rf'$|\psi_{{n={n}}}(x, y)|^2$: Tricontourf')
plt.colorbar(label=r'$|\psi|^2$')
plt.axis('equal')
plt.tight_layout()


plt.show()
