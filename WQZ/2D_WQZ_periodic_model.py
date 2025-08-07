import pybinding as pb, matplotlib.pyplot as plt, numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import math

pb.pltutils.use_style()

# Global variables:
current_M = None
shared_Wavefunction_map = None  

def QWZ_Model(t = 1, M = 2.4, a = 0.2, b = 1.5 * 0.2):

    i = 1j # Pauli matrices 
    sigma_0 = np.array([[1, 0], [0, 1]]); sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -i], [i, 0]]); sigma_z = np.array([[1, 0], [0, -1]])

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

rad = 1
shape = pb.circle(radius=rad, center=[0, 0])

scale = 5; steps = scale * 400; cap = scale * 2 * np.pi
kx = np.linspace(-cap, cap, steps) #, endpoint=False)
ky = np.linspace(-cap, cap, steps) #, endpoint=False)

M = 4.7; M_values = np.linspace(-M, M, 12)

# The wavefunction in k-space 
def compute_k_wavefunction(i):

    model = pb.Model(QWZ_Model(M = current_M), shape, pb.translational_symmetry(a1 = rad, a2 = rad))
    solver = pb.solver.lapack(model)

    k = np.size(solver.eigenvalues) # number of eigenvalues / functions

    row = np.zeros((steps, k), dtype=complex)

    for j in range (steps):
        k_point = [kx[i], ky[j]]
        k_space_function = solver.set_wave_vector(k = k_point)
        wavefunction_kx_ky = solver.eigenvectors
        row[j] = wavefunction_kx_ky[:, 0] # pick a band here 
        
    return i, row  


def compute_wavefunction_map(M):

    global current_M
    current_M = M

    # Multiprocessing
    with Pool(processes=cpu_count()) as pool:
         results = list(tqdm(pool.imap(compute_k_wavefunction, range(steps)), total=steps, desc=f"M={M} Wavefunctions"))

    model = pb.Model(QWZ_Model(M = current_M), shape, pb.translational_symmetry(a1 = rad, a2 = rad))
    solver = pb.solver.lapack(model)

    k = np.size(solver.eigenvalues) # number of eigenvalues / functions

    Wavefunction_map = np.zeros((steps, steps, k), dtype=complex)

    for i, row in results:
        Wavefunction_map[i] = row

    Wavefunction_map /= np.sum(Wavefunction_map)

    return Wavefunction_map


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
def compute_berry_curvature(i):

    global shared_Wavefunction_map

    row = np.zeros(steps -1)

    for j in range(steps -1):

        # Energy band (n = 0 or 1)
        u = shared_Wavefunction_map[i, j, :]  # wavefunction at (kx, ky)
        ux = shared_Wavefunction_map[i+1, j, :]
        uxy = shared_Wavefunction_map[i+1, j+1, :]
        uy = shared_Wavefunction_map[i, j+1, :]

        # Wavefunction alignment - u is the current point, ux is the next point
        ux = align_phase(ux, u)
        uxy = align_phase(uxy, ux)
        uy = align_phase(uy, uxy)
        
        # Berry phase 
        U1 = np.vdot(u, ux) / np.abs(np.vdot(u, ux))      #  ⟨u(kx, ky) | u(kx+Δk, ky)⟩
        U2 = np.vdot(ux, uxy) / np.abs(np.vdot(ux, uxy))  #  ⟨u(kx+Δk, ky) | u(kx+Δk, ky+Δk)⟩
        U3 = np.vdot(uxy, uy) / np.abs(np.vdot(uxy, uy))  #  ⟨u(kx+Δk, ky+Δk) | u(kx, ky+Δk)⟩
        U4 = np.vdot(uy, u) / np.abs(np.vdot(uy, u))      #  ⟨u(kx, ky+Δk) | u(kx, ky)⟩

        phase = U1 * U2 * U3 * U4 # multiply to get the full phase: exp(iφ) 
            
        # Berry curvature
        berry_curve = np.log(phase).imag # Im{ ln[ exp(iφ) ] }

        # Correct the phase to make it continuous - avoids 2π jumps
        if j > 0:
            berry_curve = phase_correction(berry_curve, row[j - 1])

        row[j] = berry_curve # φ

    return i, row


def compute_chern_number(Wavefunction_map, M):

    global shared_Wavefunction_map
    shared_Wavefunction_map = Wavefunction_map
    
    # Multiprocessing
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(compute_berry_curvature, range(steps - 1)), total=steps - 1, desc=f"M={M} Berry")) 

    berry_curve_map = np.zeros((steps, steps))

    for i, row in results:
        berry_curve_map[i, :steps -1] = row
        
    berry_flux_total = np.sum(berry_curve_map) 
    chern = round(berry_flux_total / ((cap + cap ))) # have to round because numbers are not integers e.g. 0.99999 

    return chern, berry_curve_map

# Main loop for the plots: 
cols = 4 
rows = math.ceil(len(M_values) / cols)

fig, axs = plt.subplots(rows, cols, figsize=(5* cols, 5 * rows))
axs = axs.flatten()

for idx , M in enumerate(M_values):

    Wavefunction_map = compute_wavefunction_map(M)
    chern, berry_map = compute_chern_number(Wavefunction_map, M)

    ax = axs[idx]
    c = ax.contourf(kx, ky, berry_map, cmap='plasma')
    
    ax.set_title(f'Berry Curvature (M = {M:.1f} Chern =  {chern})')
    ax.set_xlabel('kx (1/nm)')
    ax.set_ylabel('ky (1/nm)')
    fig.colorbar(c, ax = ax)

# delete any unused plots:
for i in range(len(M_values), len(axs)): 
    fig.delaxes(axs[i])

plt.tight_layout()
plt.show()
