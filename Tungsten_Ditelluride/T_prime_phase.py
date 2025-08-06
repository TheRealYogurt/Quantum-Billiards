import pybinding as pb, numpy as np
import matplotlib.pyplot as plt
from math import sqrt

pb.pltutils.use_style()

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


# No SOC but emtries are spinors:
def monolayer_1T_WTe2_basic():
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
        ([0, 0], 'Te1', 'W1', hopping_parameter(t = 0, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        ([0, 0], 'Te2', 'W1', hopping_parameter( t = t_0AB, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 

        ([0, 0], 'Te2', 'Te1', hopping_parameter(t = t_pAB, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        ([0, 0], 'W2', 'Te1', hopping_parameter(t = -t_0AB, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 

        ([0, 0], 'Te2', 'W2', hopping_parameter(t = 0, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 

        # Between Cells Hoppings: From inside the cell to the right (outside the cell)   
        ([1, 0], 'W1', 'W1', hopping_parameter(t = t_dx, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        ([1, 0], 'W2', 'W2', hopping_parameter(t = t_dx, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 

        ([1, 0], 'Te2', 'Te2', hopping_parameter(t = t_px, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        ([1, 0], 'Te1', 'Te1', hopping_parameter(t = t_px, lambda_x = 0, lambda_y = 0, lambda_z = 0 )),

        ([1, 0], 'Te2', 'Te1', hopping_parameter(t = t_pAB, lambda_x = 0, lambda_y = 0, lambda_z = 0 )),
        
        ([1, 0], 'Te2', 'W1', hopping_parameter(t = -t_0AB, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        
        ([1, 0], 'W2', 'Te1', hopping_parameter(t = t_0AB , lambda_x = 0, lambda_y = 0, lambda_z = 0)),

        ([1, 0], 'W1', 'Te2', hopping_parameter(t = t_0ABx, lambda_x = 0, lambda_y = 0, lambda_z = 0)),  
        ([1, 0], 'W1', 'Te1', hopping_parameter(t = t_0x, lambda_x = 0, lambda_y = 0, lambda_z = 0)),  

        ([1, 0], 'Te1', 'W2', hopping_parameter(t = -t_0ABx, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        
        ([1, 0], 'W2', 'Te2', hopping_parameter(t = t_0x, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        
        # Between Cells Hoppings: From inside the cell to the bottom (outside the cell)
        ([0, -1], 'Te1', 'W1', hopping_parameter(t = 0, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        
        ([0, -1], 'W2', 'W1', hopping_parameter(t = t_dAB, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        ([1, -1], 'W2', 'W1', hopping_parameter(t = t_dAB, lambda_x = 0, lambda_y = 0, lambda_z = 0)),
        
        ([0, -1], 'Te2', 'Te2', hopping_parameter(t = t_py, lambda_x = 0, lambda_y = 0, lambda_z = 0)),

        # Between Cells Hoppings: From inside the cell to the top (outside the cell)
        ([0, 1], 'Te2', 'W2', hopping_parameter(t = 0, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        
        ([0, 1], 'Te1', 'Te1', hopping_parameter(t = t_py, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
                
        # Between Cells Hoppings: From inside the cell to the left (outside the cell)
        ([-1, 0], 'W2', 'Te2', hopping_parameter(t = -t_0x, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 

        # Between Cells Hoppings: From outside the cell to the inside(from the left)
        ([-1, 0], 'W1', 'Te1', hopping_parameter(t = -t_0x, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 

        # Between Cells Hoppings: Next cell over hoppoings(tothe left)
        ([-2, 0], 'W1', 'Te2', hopping_parameter(t = -t_0ABx, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
        ([-2, 0], 'Te1', 'W2', hopping_parameter(t = t_0ABx, lambda_x = 0, lambda_y = 0, lambda_z = 0)), 
    )

    return lat


# The absolute basic: 
def monolayer_1T_WTe2_basic_basic():
    # lattice vectors
    a = 3.477; b = 6.249; a1=[a, 0]; a2=[0, b] 

    #nsite energies d -> W, p -> Te
    mu_d = 0.74; mu_p = -1.75
    
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

plt.figure(); lattice = monolayer_1T_WTe2_SOC(); lattice.plot() # plot he actuale lattice with the hoppings 

# High symmetry points for a rectangular lattice: 
a = 3.477; b = 6.249; a1=[a, 0]; a2=[0, b] 
Gamma = np.array([0, 0])    # Gamma Point
X = np.array([np.pi/a, 0])  # X Point
Y = np.array([0, np.pi/b])  # Y Point
M = np.array([np.pi/a, np.pi/b])  # M Point

fig, axs = plt.subplots(2, 1, figsize=(4, 5)) # set the single figure band plots

# First subplot: Sping orbit coupling model 
plt.sca(axs[0]) # set the axis 
model_soc = pb.Model(monolayer_1T_WTe2_SOC(), pb.translational_symmetry())
solver_soc = pb.solver.lapack(model_soc)
bands_soc = solver_soc.calc_bands(Gamma, X, Y, M)
bands_soc.plot(point_labels=[r'$\Gamma$', 'X', 'Y', 'M'])
axs[0].set_title("Band Structure: Spin Orbit Coupling")

# Second subplot: Basic model
plt.sca(axs[1]) # set the axis
model_basic = pb.Model(monolayer_1T_WTe2_basic(), pb.translational_symmetry())
solver_basic = pb.solver.lapack(model_basic)
band_basic = solver_basic.calc_bands(Gamma, X, Y, M)
band_basic.plot(point_labels=[r'$\Gamma$', 'X', 'Y', 'M'])
axs[1].set_title("Band Structure: Basic")

plt.tight_layout(); plt.show()