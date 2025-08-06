import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from math import pi, sqrt

i = 1j # Pauli matrices 
sigma_0 = np.array([[1, 0], [0, 1]]); sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -i], [i, 0]]); sigma_z = np.array([[1, 0], [0, -1]])

def QWZ_Model(t = 1, M = 1, a = 0.2, b = 1.5 * 0.2):

    # Lattice vectors:
    a1=[a, 0]; a2=[0, b]  

    lat = pb.Lattice( a1, a2) # Rectangular 2D lattice

    def onsite(M):
        M = np.multiply(M, sigma_z)
        return M
    
    def hopping(t, sigma):
        sigma = np.multiply(sigma, i)
        pauli = np.subtract(sigma_z, sigma,)
        parameter = np.multiply(t/2,pauli)
        return parameter 

    lat.add_sublattices(
        # Main cell: 
        ('A', [ -0.25 * a, 0.32* b], onsite(M = M))
    )
    
    lat.add_hoppings( 
        ([1, 0], 'A', 'A', hopping(t = t, sigma = sigma_x)),
        ([0, 1], 'A', 'A', hopping(t = t, sigma = sigma_y))
    )

    return lat

scale = 1; x = 2 * scale; y = 2 * scale ; rad = 3 * scale 

circle1 = pb.circle(radius= rad, center=[x,y]); circle2 = pb.circle(radius= rad, center=[-x,-y]) 
circle3 = pb.circle(radius= rad, center=[-x,y]); circle4 = pb.circle(radius= rad, center=[x,-y])
circle5 = pb.circle(radius=rad*0.25, center=[-x/2, y/4])

shape = circle1 + circle2 + circle3 + circle4 - circle5; #plt.figure(); shape.plot()

M = 4; M_values = np.linspace(-M, M, 2*M)

# sub-plot sructure
n_cols = 4
n_total_plots = 2 * len(M_values)  # 2 plots per M
n_rows = int(np.ceil(n_total_plots / n_cols))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axs = axs.flatten()

# Loop over the different M values
for idx, M in enumerate(M_values):

    # Create model for each M
    model = pb.Model(QWZ_Model(M=M), shape)
    solver = pb.solver.arpack(model, k=350 * scale)

    # Eigenvalues and spectral staircase
    eigenvalues = solver.calc_eigenvalues()
    Sorted_Eigenvalues = np.sort(solver.eigenvalues)
    N_E = np.arange(1, len(Sorted_Eigenvalues) + 1)

    # Polynomial fit
    poly = Polynomial.fit(Sorted_Eigenvalues, N_E, deg=37)
    N_smooth = poly(Sorted_Eigenvalues)
    S_n_2 = np.diff(N_smooth)
    S_n_normalised_2 = S_n_2 / np.mean(S_n_2)

    # Set up binning
    s = np.linspace(0, 6, 100000)
    Bin = round(np.sqrt(len(Sorted_Eigenvalues)) * 0.6)

    # Plot Spectral Staircase - left plot 
    if 2 * idx < len(axs): 
        ax = axs[2 * idx]  
        ax.step(Sorted_Eigenvalues, N_E, label="Numerical $N(E)$", where='post')
        ax.plot(Sorted_Eigenvalues, N_smooth, label="Smoothed $N(E)$", linestyle='dashed')

        ax.set_xlabel("E")
        ax.set_ylabel("N(E)")
        ax.set_title(f"Spectral Staircase Function (M = {M:.1f})")
        ax.legend(); ax.grid()

    # Plot Level Spacing Distribution - right plot
    if 2 * idx + 1 < len(axs):  
        ax = axs[2 * idx + 1]  
        ax.hist(S_n_normalised_2, bins=Bin, density=True, alpha=0.6, label="Numerical $P(S)$")

        ax.plot(s, np.exp(-s), linestyle='dashed', label="Poisson")
        ax.plot(s, (np.pi / 2) * s * np.exp(- (np.pi / 4) * s**2), linestyle='solid', label="GOE")
        ax.plot(s, (32 / (np.pi**2)) * s**2 * np.exp(- (4 / np.pi) * s**2), linestyle='dotted', label="GUE")

        ax.set_xlabel("Unfolded Level Spacing $S$")
        ax.set_ylabel("Probability $P(S)$")
        ax.set_title(f"Level Spacing Distribution (M = {M:.1f})")
        ax.legend(); ax.grid()


fig.suptitle("5 Circles Billiard for Different M Values", fontsize=18, y=0.99)  
fig.tight_layout(rect=[0, 0, 1, 0.99])  
plt.show()  # spit out the plots

