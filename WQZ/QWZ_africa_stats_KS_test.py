import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from math import pi, sqrt

from scipy.stats import kstest
from scipy.integrate import quad

i = 1j 
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

def p_poisson(s):
    poisson = np.exp(-s)
    return poisson

def cdf_poisson(s):
    result, _ = quad(p_poisson, 0, s)
    return result
vec_cdf_poisson = np.vectorize(cdf_poisson)

# GOE PDF and CDF
def p_goe(s):
    goe = (np.pi / 2) * s * np.exp(- (np.pi / 4) * s**2)
    return goe

def cdf_goe(s):
    result, _ = quad(p_goe, 0, s)
    return result
vec_cdf_goe = np.vectorize(cdf_goe)

# GUE PDF and CDF
def p_gue(s):
    gue = (32 / pi**2) * s**2 * np.exp(-4 * s**2 / pi)
    return gue

def cdf_gue(s):
    result, _ = quad(p_gue, 0, s)
    return result
vec_cdf_gue = np.vectorize(cdf_gue)

scale = 2; x0 = 1.1 * scale; y0 = 0.5 * scale; shifty = 1.44 * scale; shiftx = -0.6 * scale

circle1 = pb.circle(radius=2 * scale, center=[0 * scale, 0 * scale])
circle2 = pb.circle(radius=1.31 * scale, center=[-1.7 * scale, 0.66 * scale]) 
rectangle = pb.Polygon([[x0 + shiftx, y0 + shifty], [x0 + shiftx, -y0 + shifty], [-x0 + shiftx, -y0 + shifty], [-x0 + shiftx, y0 + shifty]])

shape = circle1 + circle2 + rectangle

M_values = [-1.7, 0, 1.7]  

# 3 plots per M
n_cols = 3
n_total_plots = 3 * len(M_values)  
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
    poly = Polynomial.fit(Sorted_Eigenvalues, N_E, deg=101)
    N_smooth = poly(Sorted_Eigenvalues)
    S_n_2 = np.diff(N_smooth)
    S_n_normalised_2 = S_n_2 / np.mean(S_n_2)

    residuals = N_E - N_smooth
    rmse = np.sqrt(np.mean(residuals**2))

    D_poisson_, p_poisson_ = kstest(S_n_normalised_2, vec_cdf_poisson) # KS test vs GOE
    D_goe_, p_goe_ = kstest(S_n_normalised_2, vec_cdf_goe) # KS test vs GOE
    D_gue_, p_gue_ = kstest(S_n_normalised_2, vec_cdf_gue) # KS test vs GUE
    
    # Binning
    s = np.linspace(0, 6, 100000)
    Bin = round(np.sqrt(len(Sorted_Eigenvalues))*0.65)

    # Spectral Staircase
    ax = axs[3 * idx]
    ax.step(Sorted_Eigenvalues, N_E, label="Numerical $N(E)$", where='post')
    ax.plot(Sorted_Eigenvalues, N_smooth, label="Smoothed $N(E)$", linestyle='dashed')
    ax.set_xlabel("E"); ax.set_ylabel("N(E)")
    ax.set_title(f"Staircase (M = {M:.1f})")
    ax.legend(); ax.grid()

    # Level Spacing Distribution
    ax = axs[3 * idx + 1]
    ax.hist(S_n_normalised_2, bins=Bin, density=True, alpha=0.6, label="Numerical $P(S)$")
   
    label_poisson = fr"Poisson$_{{\mathit{{D}}={D_poisson_:.3f},\ \mathit{{p}}={p_poisson_:.3f}}}$"
    label_goe = fr"GOE$_{{\mathit{{D}}={D_goe_:.3f},\ \mathit{{p}}={p_goe_:.3f}}}$"
    label_gue = fr"GUE$_{{\mathit{{D}}={D_gue_:.3f},\ \mathit{{p}}={p_gue_:.3f}}}$"

    ax.plot(s, np.exp(-s), linestyle='dashed', label=label_poisson)
    ax.plot(s, (np.pi / 2) * s * np.exp(- (np.pi / 4) * s**2), linestyle='solid', label=label_goe)
    ax.plot(s, (32 / (np.pi**2)) * s**2 * np.exp(- (4 / np.pi) * s**2), linestyle='dotted', label=label_gue)

    ax.set_xlabel("Unfolded Level Spacing $S$")
    ax.set_ylabel("Probability $P(S)$")
    ax.set_title(f"Spacing (M = {M:.1f})")
    ax.legend(); ax.grid()

    # Residuals
    ax = axs[3 * idx + 2]
    ax.plot(Sorted_Eigenvalues, residuals, lw=1)
    ax.axhline(0, color="black", linestyle="dashed", lw=1)
    ax.set_xlabel("E")
    ax.set_ylabel("Residuals $N(E) - N_{smooth}(E)$")
    ax.set_title(f"Residuals (M = {M:.1f}, RMSE={rmse:.3f})")
    ax.grid()

fig.suptitle("Africa Billiard for Different M Values", fontsize=18, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
