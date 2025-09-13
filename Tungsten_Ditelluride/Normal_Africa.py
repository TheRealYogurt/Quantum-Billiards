import pybinding as pb, matplotlib.pyplot as plt, numpy as np 
from pybinding.repository import group6_tmd
from math import pi, sqrt
from numpy.polynomial import Polynomial

#################################### THE SYSTEM #################################### 
scale = 1; x0 = 1.1 * scale; y0 = 0.5* scale; shifty = 1.44* scale; shiftx = -0.6* scale
circle1 =  pb.circle(radius=2* scale) # define a circle with radius 1nm
circle2 = pb.circle(radius=1.31* scale, center=[-1.7* scale, 0.66* scale]) # define a circle with radius 1nm
rectangle = pb.Polygon([[x0 + shiftx, y0+ shifty], [x0+ shiftx, -y0+ shifty], [-x0+ shiftx, -y0+ shifty], [-x0+ shiftx, y0+ shifty]])
shape = circle1 + circle2 + rectangle; #shape.plot()

model = pb.Model(group6_tmd.monolayer_3band("WTe2"),shape) #plt.figure(figsize=(6,4)); model.plot() # define the model and plot it 
#################################### THE SYSTEM ####################################

#################################### SOLVE THE EIGENVALUE PROBLEM ####################################
solver = pb.solver.arpack(model, k=425 * scale, sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 
eigenvalues = solver.calc_eigenvalues(); #plt.figure(figsize=(6,4)); eigenvalues.plot()
Sorted_Eigenvalues = np.sort(solver.eigenvalues) # sort the eigen values in ascending order, to ensure lowest ones come first  
N_E = np.arange(1, len(Sorted_Eigenvalues) + 1)  # Number of states below each energy level
#################################### SOLVE THE EIGENVALUE PROBLEM ####################################

poly = Polynomial.fit(Sorted_Eigenvalues, N_E, deg=35)
N_smooth = poly(Sorted_Eigenvalues) # extracting the smoothed over Eigenvalues
S_n_2 = np.diff(N_smooth); S_n_normalised_2 = S_n_2/np.mean(S_n_2) # Compute spacings between each level

s = np.linspace(0, 6, 100000) # create an array to plot perfect model
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
Bin =round(np.sqrt(len(Sorted_Eigenvalues)))

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
