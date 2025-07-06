import pybinding as pb, matplotlib.pyplot as plt, numpy as np

pb.pltutils.use_style()

# Pauli matrices 
i = 1j 
sigma_0 = np.array([[1, 0], [0, 1]]); sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -i], [i, 0]]); sigma_z = np.array([[1, 0], [0, -1]])

def QWZ_Model(t, M, a, b):


    # Lattice vectors:
    a1=[a, 0]; a2=[0, b]  

    lat = pb.Lattice( a1, a2) # Rectangular 2D lattice

    def onsite(M):
        M = np.multiply(M, sigma_z)
        return M
    
    def hopping(t, sigma):
        sigma = np.multiply(sigma, i)
        pauli = np.subtract(sigma_z, sigma,)
        parameter = np.multiply(-t/2,pauli)
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
#plt.figure(); lattice = QWZ_Model(); lattice.plot()

# Constants in the model / Lattice size:
d = 0.2; a = d; b = 1.5*d
t = 1

# High symmetry points for a rectangular lattice
Gamma = np.array([0, 0])
X = np.array([np.pi/a, 0])
Y = np.array([0, np.pi/b])
M_point = np.array([np.pi/a, np.pi/b])


cap = 4
M_list = np.linspace(-cap, cap, 2*cap+1)

# Create subplots: 1 row per M, 2 columns (left=linear, right=high symmetry)
fig, axs = plt.subplots(len(M_list), 2, figsize=(6, 1 * len(M_list)), sharex='col')

# Loop over each M value
for idx, M_val in enumerate(M_list):
    # Create the model and solver
    model = pb.Model(QWZ_Model(t=t, M=M_val, a=a, b=b), pb.translational_symmetry())
    solver = pb.solver.lapack(model)

    #Linear Path Plot (Left)
    bands_linear = solver.calc_bands(-X, Gamma, X)
    ax_left = axs[idx, 0]
    plt.sca(ax_left)
    bands_linear.plot(point_labels=['-X', r'$\Gamma$', 'X'])
    ax_left.set_title(f"Linear Path (M = {M_val})")

    # High Symmetry Path Plot (Right)
    bands_highsym = solver.calc_bands(Gamma, X, Y, M_point)
    ax_right = axs[idx, 1]
    plt.sca(ax_right)
    bands_highsym.plot(point_labels=[r'$\Gamma$', 'X', 'Y', 'M'])
    ax_right.set_title(f"High Symmetry Path (M = {M_val})")

plt.tight_layout(); plt.show()



""" 
M = 3
model = pb.Model(QWZ_Model(t = t, M = M, a = a, b = b), pb.translational_symmetry())
solver = pb.solver.lapack(model)

# High symmetry points for a rectangular lattice: 
Gamma = np.array([0, 0])    # Gamma Point
X = np.array([np.pi/a, 0])  # X Point
Y = np.array([0, np.pi/b])  # Y Point
M = np.array([np.pi/a, np.pi/b])  # M Point
 
fig, axs = plt.subplots(2, 1, figsize=(4, 5)) # set the single figure band plots

plt.sca(axs[0]) # set the axis
bands = solver.calc_bands(Gamma, X, Y, M)
bands.plot(point_labels=[r'$\Gamma$', 'X', 'Y', 'M'])
axs[0].set_title("Band Structure: Rectangle Symmetry points")

#linear path
plt.sca(axs[1]) # set the axis
point_1 = - np.array([np.pi/a, 0])
point_2 = - point_1
bands = solver.calc_bands(point_1, Gamma, point_2)
bands.plot(point_labels=[r'$-\pi$', r'$\Gamma$', r'$\pi$'])
axs[1].set_title("Band Structure: -π to π")

plt.tight_layout(); plt.show()
"""