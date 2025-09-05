import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from math import pi, sqrt
from scipy.stats import chisquare   # <-- added

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

scale = 1; x0 = 1.1 * scale; y0 = 0.5 * scale; shifty = 1.44 * scale; shiftx = -0.6 * scale

circle1 = pb.circle(radius=2 * scale, center=[0 * scale, 0 * scale])
circle2 = pb.circle(radius=1.31 * scale, center=[-1.7 * scale, 0.66 * scale]) 
rectangle = pb.Polygon([[x0 + shiftx, y0 + shifty], [x0 + shiftx, -y0 + shifty], [-x0 + shiftx, -y0 + shifty], [-x0 + shiftx, y0 + shifty]])

shape = circle1 + circle2 + rectangle

M_values = [-1.7, 0, 1.7]  

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
    Bin = round(np.sqrt(len(Sorted_Eigenvalues)) * 0.65)

    # === Histogram for χ² ===
    counts, bin_edges = np.histogram(S_n_normalised_2, bins=Bin, range=(0,6), density=False)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    total = np.sum(counts)

    # Expected distributions (rescaled to histogram counts)
    pdf_poisson = np.exp(-bin_centers)
    pdf_goe = (np.pi / 2) * bin_centers * np.exp(-(np.pi / 4) * bin_centers**2)
    pdf_gue = (32 / (np.pi**2)) * bin_centers**2 * np.exp(-(4 / np.pi) * bin_centers**2)

    # Normalize to expected frequencies
    exp_poisson = pdf_poisson / np.sum(pdf_poisson) * total
    exp_goe = pdf_goe / np.sum(pdf_goe) * total
    exp_gue = pdf_gue / np.sum(pdf_gue) * total

    # Chi-square tests
    chi2_pois, p_pois = chisquare(f_obs=counts, f_exp=exp_poisson)
    chi2_goe, p_goe = chisquare(f_obs=counts, f_exp=exp_goe)
    chi2_gue, p_gue = chisquare(f_obs=counts, f_exp=exp_gue)

    # Build labels with chi² and p values in subscript
    label_poisson = fr"Poisson$_{{\chi^2={chi2_pois:.2f},\ p={p_pois:.3f}}}$"
    label_goe     = fr"GOE$_{{\chi^2={chi2_goe:.2f},\ p={p_goe:.3f}}}$"
    label_gue     = fr"GUE$_{{\chi^2={chi2_gue:.2f},\ p={p_gue:.3f}}}$"

    # --- Plot spectral staircase ---
    if 2 * idx < len(axs): 
        ax = axs[2 * idx]  
        ax.step(Sorted_Eigenvalues, N_E, label="Numerical $N(E)$", where='post')
        ax.plot(Sorted_Eigenvalues, N_smooth, label="Smoothed $N(E)$", linestyle='dashed')

        ax.set_xlabel("E"); ax.set_ylabel("N(E)")
        ax.set_title(f"Spectral Staircase (M = {M:.1f})")
        ax.legend(); ax.grid()

    # --- Plot Level Spacing Distribution ---
    if 2 * idx + 1 < len(axs):  
        ax = axs[2 * idx + 1]  
        ax.hist(S_n_normalised_2, bins=Bin, density=True, alpha=0.6, label="Numerical $P(S)$")

        ax.plot(s, np.exp(-s), linestyle='dashed', label=label_poisson)
        ax.plot(s, (np.pi / 2) * s * np.exp(- (np.pi / 4) * s**2), linestyle='solid', label=label_goe)
        ax.plot(s, (32 / (np.pi**2)) * s**2 * np.exp(- (4 / np.pi) * s**2), linestyle='dotted', label=label_gue)

        ax.set_xlabel("Unfolded Level Spacing $S$")
        ax.set_ylabel("Probability $P(S)$")
        ax.set_title(f"Level Spacing Distribution (M = {M:.1f})")
        ax.legend(); ax.grid()

fig.suptitle("Africa Billiard for Different M Values", fontsize=18, y=0.99)  
fig.tight_layout(rect=[0, 0, 1, 0.99])  
plt.show() # spit out the plots
