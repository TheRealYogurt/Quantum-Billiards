import pybinding as pb, matplotlib.pyplot as plt, numpy as np 
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial


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

# plt.figure(); lattice = monolayer_1T_WTe2_SOC(); lattice.plot()


scale = 9
x = 2 * scale; y = 2 * scale ; rad = 3 * scale 
circle1 = pb.circle(radius= rad, center=[x,y]); circle2 = pb.circle(radius= rad, center=[-x,-y]) 
circle3 = pb.circle(radius= rad, center=[-x,y]); circle4 = pb.circle(radius= rad, center=[x,-y])

shape = circle1 + circle2 + circle3 + circle4;  #shape.plot()

model = pb.Model(monolayer_1T_WTe2_SOC(), shape)
solver = pb.solver.arpack(model, k=400 , sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 


eigenvalues = solver.calc_eigenvalues() 
Sorted_Eigenvalues = np.sort(solver.eigenvalues) # sort the eigen values in ascending order, to ensure lowest ones come first  
N_E = np.arange(1, len(Sorted_Eigenvalues) + 1)  # Number of states below each energy level

poly = Polynomial.fit(Sorted_Eigenvalues, N_E, deg=41)
N_smooth = poly(Sorted_Eigenvalues) # extracting the smoothed over Eigenvalues
S_n_2 = np.diff(N_smooth); S_n_normalised_2 = S_n_2/np.mean(S_n_2) # Compute spacings between each level

s = np.linspace(0, 6, 100000) # create an array to plot perfect model
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
Bin = round(np.sqrt(len(Sorted_Eigenvalues)) * 0.55)


ax = axs[0]  
ax.step(Sorted_Eigenvalues, N_E, label="Numerical $N(E)$", where='post')
ax.plot(Sorted_Eigenvalues, N_smooth, label="Smoothed $N(E)$", linestyle='dashed')
ax.set_xlabel("E"); ax.set_ylabel("N(E)"); ax.set_title("Spectral Staircase Function")
ax.legend(); ax.grid()

ax = axs[1] 
ax.hist(S_n_normalised_2, bins=Bin, density=True, alpha=0.6, label="Numerical $P(S)$")
ax.plot(s, np.exp(-s), linestyle='dashed', label="Poisson") # Poisson distribution
ax.plot(s, (np.pi / 2) * s * np.exp(- (np.pi / 4) * s**2), linestyle='solid', label="GOE") # GOE distribution 
ax.plot(s, (32 / (np.pi**2)) * s**2 * np.exp(- (4 / np.pi) * s**2), linestyle='dotted', label="GUE") # GUE distribution
#ax.plot(s, ((2**18)/(3**6 * pi**3))*s**4*np.exp(-(64/(9*pi))*s**2), linestyle='dashdot', label="GSE") # GSE distribution 
ax.set_xlabel("Unfolded Level Spacing $S$"); ax.set_ylabel("Probability $P(S)$"); ax.set_title("Level Spacing Distribution")
ax.legend(); ax.grid()

plt.show() # spit out the plots