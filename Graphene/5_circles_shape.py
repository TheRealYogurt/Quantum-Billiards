import pybinding as pb, matplotlib.pyplot as plt, numpy as np 
from pybinding.repository import graphene
from math import pi, sqrt
from numpy.polynomial import Polynomial

scale = 2; x = 2 * scale; y = 2 * scale ; rad = 3 * scale 



circle1 = pb.circle(radius= rad, center=[x,y]); circle2 = pb.circle(radius= rad, center=[-x,-y]) 
circle3 = pb.circle(radius= rad, center=[-x,y]); circle4 = pb.circle(radius= rad, center=[x,-y])
circle5 = pb.circle(radius=rad*0.25, center=[-x/2, y/4])


shape = circle1 + circle2 + circle3 + circle4 - circle5; #plt.figure(); shape.plot()


model = pb.Model(graphene.monolayer(),shape); #model.plot()

#################################### SOLVE THE EIGENVALUE PROBLEM ####################################
solver = pb.solver.arpack(model, k=520 * scale, sigma=0.2) # solves for the k-number lowest energy eigen values around sigma 
eigenvalues = solver.calc_eigenvalues(); #plt.figure(figsize=(6,4)); eigenvalues.plot()
Sorted_Eigenvalues = np.sort(solver.eigenvalues) # sort the eigen values in ascending order, to ensure lowest ones come first  
N_E = np.arange(1, len(Sorted_Eigenvalues) + 1)  # Number of states below each energy level
#################################### SOLVE THE EIGENVALUE PROBLEM ####################################

#################################### POLYNOMIAL FITTING ####################################
poly = Polynomial.fit(Sorted_Eigenvalues, N_E, deg=101)
N_smooth = poly(Sorted_Eigenvalues) # extracting the smoothed over Eigenvalues
S_n_2 = np.diff(N_smooth); S_n_normalised_2 = S_n_2/np.mean(S_n_2) # Compute spacings between each level
#################################### POLYNOMIAL FITTING ####################################

#------------- SETTING UP 2x2 GRID FOR ALL PLOTS ----------------------#
s = np.linspace(0, 6, 100000) # create an array to plot perfect model
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
Bin = round(sqrt(len(Sorted_Eigenvalues)) *0.75)
#------------- SETTING UP 2x2 GRID FOR ALL PLOTS ----------------------#
 
#-------------PLOT 3: Polynomial Fit - Spectral Staircase----------------------#
ax = axs[0]  # bottom-left
ax.step(Sorted_Eigenvalues, N_E, label="Numerical $N(E)$", where='post')
ax.plot(Sorted_Eigenvalues, N_smooth, label="Smoothed $N(E)$", linestyle='dashed')
ax.set_xlabel("E"); ax.set_ylabel("N(E)"); ax.set_title("Spectral Staircase Function - Polynomial Fit")
ax.legend(); ax.grid()

#-------------PLOT 4: Polynomial Fit - Level Spacing Distribution----------------------#
ax = axs[1]  # bottom-right
ax.hist(S_n_normalised_2, bins=Bin, density=True, alpha=0.6, label="Numerical $P(S)$")
ax.plot(s, np.exp(-s), linestyle='dashed', label="Poisson") # Poisson distribution
ax.plot(s, (pi / 2) * s * np.exp(- (np.pi / 4) * s**2), linestyle='solid', label="GOE") # GOE distribution 
ax.plot(s, (32 / (np.pi**2)) * s**2 * np.exp(- (4 / np.pi) * s**2), linestyle='dotted', label="GUE") # GUE distribution
#ax.plot(s, ((2**18)/(3**6 * pi**3))*s**4*np.exp(-(64/(9*pi))*s**2), linestyle='dashdot', label="GSE") # GSE distribution 
ax.set_xlabel("Unfolded Level Spacing $S$"); ax.set_ylabel("Probability $P(S)$"); ax.set_title("Level Spacing Distribution- Polynomial Fit")
ax.legend(); ax.grid()


plt.show() # spit out the plots