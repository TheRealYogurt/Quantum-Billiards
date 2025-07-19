import pybinding as pb, matplotlib.pyplot as plt, numpy as np
from numpy.linalg import eigh

pb.pltutils.use_style()

# Pauli matrices 
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
#plt.figure(); lattice = QWZ_Model(); lattice.plot()

# Constants in the model / Lattice size:
d = 0.2; a = d; b = 1.5*d
t = 1; M = 4.7 # Onsite energy

""" Will produce multiple plots for the ban structure for different values of M (from 'cap' to '-cap'): 
# High symmetry points for a rectangular lattice
Gamma = np.array([0, 0])
X = np.array([np.pi/a, 0])
Y = np.array([0, np.pi/b])
M_point = np.array([np.pi/a, np.pi/b])

cap = round(M)
M_list = np.linspace(-cap, cap, 2*cap+1)

# Create subplots: 1 row per M, 2 columns (left=linear, right=high symmetry)
fig, axs = plt.subplots(len(M_list), 2, figsize=(6, 1 * len(M_list)), sharex='col')

# Loop over each M value
for idx, M_val in enumerate(M_list):
    # Create the model and solver
    model = pb.Model(QWZ_Model(t=t, M=M_val, a=a, b=b), pb.translational_symmetry())
    solver = pb.solver.lapack(model)

    # High Symmetry Path Plot - Negative (Left)
    bands_linear = solver.calc_bands(-M_point, -Y, -X, Gamma)
    ax_left = axs[idx, 0]
    plt.sca(ax_left)
    bands_linear.plot(point_labels=['-M', '-Y', '-X', r'$\Gamma$'])
    ax_left.set_title(f"High Symmetry Path - Negative (M = {M_val})")

    # High Symmetry Path Plot - Positive (Right)
    bands_highsym = solver.calc_bands(Gamma, X, Y, M_point)
    ax_right = axs[idx, 1]
    plt.sca(ax_right)
    bands_highsym.plot(point_labels=[r'$\Gamma$', 'X', 'Y', 'M'])
    ax_right.set_title(f"High Symmetry Path - Positive (M = {M_val})")

plt.tight_layout()#; plt.show()
#"""

""" Will produce a single figure with 2 subplots for M - negative and positive k paths
model = pb.Model(QWZ_Model(t = t, M = M, a = a, b = b), pb.translational_symmetry())
solver = pb.solver.lapack(model)

# High symmetry points for a rectangular lattice: 
Gamma = np.array([0, 0])    # Gamma Point
X = np.array([np.pi/a, 0])  # X Point
Y = np.array([0, np.pi/b])  # Y Point
M = np.array([np.pi/a, np.pi/b])  # M Point
 
fig, axs = plt.subplots(1, 2, figsize=(5, 2)) # set the single figure band plots

# Negative (Left)
plt.sca(axs[0]) # set the axis
bands = solver.calc_bands(-M, -Y, -X, Gamma)
bands.plot(point_labels=['-M', '-Y', '-X', r'$\Gamma$'])
axs[0].set_title("High Symmetry points - Negative", fontsize = 6)

# Positive (Righ)
plt.sca(axs[1]) # set the axis
bands = solver.calc_bands(Gamma, X, Y, M)
bands.plot(point_labels=[r'$\Gamma$', 'X', 'Y', 'M'])
axs[1].set_title("High Symmetry points - Positive", fontsize = 6)

plt.tight_layout()
"""


# QWZ Bloch Hamiltonian
def H_k(kx, ky, M, t): 
    dx = t * np.sin(kx)
    dy = t * np.sin(ky)
    dz = M + t * np.cos(kx) + t * np.cos(ky)
    return dx * sigma_x + dy * sigma_y + dz * sigma_z 

scale = 1; cap = scale * np.pi

# Makes the berry phase continuous by removing 2pi jumps
def phase_correction(phase_current, phase_previous):
    if phase_current - phase_previous > np.pi:
        return phase_current - 2 * np.pi
    elif phase_current - phase_previous < -np.pi:
            return phase_current + 2 * np.pi
    else:   
        return phase_current

# Aligns the phase of the current wavefunction to a reference wavefunction
def align_phase(current, reference):
    phase = np.vdot(reference, current)
    return current * np.exp(-1j * np.angle(phase))

# The Chern number calculation: 
def compute_chern_number(M, t, band, Nk=scale * 100):
    kx_vals = np.linspace(-cap, cap, Nk) # grid points for the x - axis values 
    ky_vals = np.linspace(-cap, cap, Nk) # grid points for the y - axis values  

    berry_flux_total = 0.0 # Initial value of the integral
    
    berry_curve_map = np.zeros((Nk, Nk))

    for i in range(Nk):
        for j in range(Nk):
            # for a square in the BZ
            kx = kx_vals[i]
            ky = ky_vals[j]
            kx_dx = kx_vals[(i + 1) % Nk]
            ky_dy = ky_vals[(j + 1) % Nk]

            # Solve and get eigenvectors of a band n        
            _, v = eigh(H_k(kx, ky, M, t))
            _, vx = eigh(H_k(kx_dx, ky, M, t))
            _, vxy = eigh(H_k(kx_dx, ky_dy, M, t))
            _, vy = eigh(H_k(kx, ky_dy, M, t))

            # Energy band (n = 0 or 1)
            u = v[:, band]
            ux = vx[:, band]
            uxy = vxy[:, band]
            uy = vy[:, band]

            # Wavefunction alignment - u is the current point, ux is the next point
            ux = align_phase(ux, u)
            uxy = align_phase(uxy, ux)
            uy = align_phase(uy, uxy)
            
            """ 
            (kx, ky+Δk)   --->   (kx+Δk, ky+Δk)
                ^                         ^
                |                         |
            (kx, ky)      --->      (kx+Δk, ky)
            """

            # Berry phase 
            U1 = np.vdot(u, ux) / np.abs(np.vdot(u, ux))      #  ⟨u(kx, ky) | u(kx+Δk, ky)⟩
            U2 = np.vdot(ux, uxy) / np.abs(np.vdot(ux, uxy))  #  ⟨u(kx+Δk, ky) | u(kx+Δk, ky+Δk)⟩
            U3 = np.vdot(uxy, uy) / np.abs(np.vdot(uxy, uy))  #  ⟨u(kx+Δk, ky+Δk) | u(kx, ky+Δk)⟩
            U4 = np.vdot(uy, u) / np.abs(np.vdot(uy, u))      #  ⟨u(kx, ky+Δk) | u(kx, ky)⟩

            phase = U1 * U2 * U3 * U4 # multiply to get the full phase: exp(iφ) 
            
            # Berry curvature
            berry_curve = np.log(phase).imag # Im{ ln[ exp(iφ) ] }

            # Correct the phase to make it continuous - avoids 2π jumps
            if i > 0 and j > 0:
                berry_curve = phase_correction(berry_curve, berry_curve_map[i - 1, j - 1])


            berry_curve_map[i,j] = berry_curve # φ

            berry_flux_total += berry_curve # φ

            chern = round(berry_flux_total / (scale * (cap + cap ))) # have to round because numbers are not integers e.g. 0.99999 

    return  chern, berry_curve_map

""" 
chern, berry_map = compute_chern_number(M = M, t = t, band = 0)
print("For M = ", M, ", Chern number:", chern)

kx_vals = np.linspace(-cap, cap, scale * 100) # grid points for the x - axis values 
ky_vals = np.linspace(-cap, cap, scale * 100) # grid points for the y - axis values

plt.figure(figsize=(6,5))
plt.contourf(kx_vals, ky_vals, berry_map.T, cmap= 'plasma')
plt.colorbar()
plt.title(f'Berry Curvature in BZ (Chern = {chern})')
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.tight_layout()
plt.show()
"""


M_list = np.linspace(-M, M, 12) 
num_plots = len(M_list)

fig, axes = plt.subplots(3, 4, figsize=(10, 6))  # 3 rows, 4 columns

for i in range(num_plots):
    chern, berry_map = compute_chern_number(M = M_list[i], t=t, band=0)
    
    ax = axes[i // 4, i % 4]  
    
    kx_vals = np.linspace(-cap, cap, scale * 100) # grid points for the x - axis values 
    ky_vals = np.linspace(-cap, cap, scale * 100) # grid points for the y - axis values
    c = ax.contourf(kx_vals, ky_vals, berry_map.T, cmap= 'plasma') # countourf plot fills the area with colors
    
    ax.set_title(f'Berry Curvature (M={M_list[i]:.1f}, Chern={chern})', fontsize=8)
    ax.set_xlabel('$k_x$', fontsize=10)
    ax.set_ylabel('$k_y$', fontsize=10)

    fig.colorbar(c, ax=ax)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
