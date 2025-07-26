import pybinding as pb, matplotlib.pyplot as plt, numpy as np

pb.pltutils.use_style()

# Pauli matrices 
i = 1j 
sigma_0 = np.array([[1, 0], [0, 1]]); sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -i], [i, 0]]); sigma_z = np.array([[1, 0], [0, -1]])

def QWZ_Model(t = 1, M = 1, a = 0.2, b = 0.2):

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

rad = 2.1; k = int(21**2)
shape = pb.circle(radius=rad, center=(0, 0))
model =pb.Model(QWZ_Model(), shape)

# Eigenvalues and Eigenfunctions for the system in real space 
solver = pb.solver.arpack(model, k=k, sigma=0.2)
wavefunctions = solver.eigenvectors
eigenvalues = solver.eigenvalues
positions = solver.system.positions # real space positions of each atom - i think 
 
fscale = 50; cap = fscale * 2 * np.pi
r = np.column_stack(positions); steps = 8 * np.size(r[:,0])
kx = np.linspace(-cap, cap, steps)
ky = np.linspace(-cap, cap, steps)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

mesh_step = kx[1]-kx[0]

max_M_xy = np.zeros((k, 2), dtype=complex)
max_FT_psi = np.zeros((k, 2), dtype=complex) 

for n in range(len(k)):
    n = n; psi = wavefunctions[:, n] # eigen fucntion, index 0 being the fisrt
    psi_reshaped = psi.reshape((len(r), 2))  # reshape 


    M_xy = np.zeros((steps,steps),dtype=complex)
    FT_psi = np.zeros((steps, steps, 2), dtype=complex)

    # Calculate the Fourier transform of the wavefunction
    for i in range(steps):
        for j in range(steps):
            phase = np.exp(-1j * (kx[i] * r[:, 0] + ky[j] * r[:, 1]))  # shape: (num_sites,)

            # Apply the fourier transform to the wavefunction fourier phase * [psi_up, psi_down] 
            U1 =np.dot(psi_reshaped.T, phase) 

            # Sum over all orbitals to get the total transform at each k-point
            U2 = np.sum(U1)
        
            # Multiply by the mesh step size to get the density
            M_xy[i, j] = U2  * mesh_step**2 

            # Store the Fourier transform for each k-point as a spinor
            FT_psi[i, j] = U1 * mesh_step**2

        print(f"Progress: {i+1}/{steps}")

    M_xy /= np.sum(M_xy) # The normalise Momentum distribution
    idx = np.unravel_index(np.argmax(abs(M_xy)), M_xy.shape)
    max_M_xy = [n, 0], max_M_xy[n, 1] = kx[idx[0]], ky[idx[1]]


    FT_psi/= np.sum(FT_psi, axis=0) # The normalise Momentum distribution
    idy = np.unravel_index(np.argmax(np.abs(FT_psi)), FT_psi.shape)
    max_FT_psi[n] = FT_psi[idy[0], idy[1]]

# check that the eigevalues is a pefrect square for reshaping
grid_size = int(np.sqrt(k))
if grid_size ** 2 != k:
    raise ValueError("k must be a perfect square to reshape into a grid")

max_FT_psi_grid = max_FT_psi.reshape(grid_size, grid_size, 2)

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


Nk = int(np.sqrt(k))-1  # Number of k-points in each direction (excluding the last point for periodicity)

berry_flux_total = 0.0 # Initial value of the integral
    
berry_curve_map = np.zeros((Nk, Nk))

for i in range(Nk):
    for j in range(Nk):

        # Energy band (n = 0 or 1)
        u = max_FT_psi_grid[i, j]  # wavefunction at (kx, ky)
        ux = max_FT_psi_grid[i+1, j]
        uxy = max_FT_psi_grid[i+1, j+1]
        uy = max_FT_psi_grid[i, j+1]

        #u =  M_xy[i, j]  # wavefunction at (kx, ky)
        #ux =  M_xy[i+1, j]
        #uxy = M_xy[i+1, j+1]
        #uy = M_xy[i, j+1]

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
        if i > 0 and j > 0:
            berry_curve = phase_correction(berry_curve, berry_curve_map[i - 1, j - 1])


        berry_curve_map[i,j] = berry_curve # φ

        berry_flux_total += berry_curve # φ

        Chern = round( berry_flux_total / ((2 * cap) )) # have to round because numbers are not integers e.g. 0.99999 

    print(f"Progress: {i+1}/{Nk}")

print(f"Chern number: {Chern}")

# Plotting the Berry curvature map
plt.figure()
plt.contourf(kx[:-1], ky[:-1], berry_curve_map, cmap='plasma')
plt.title(f'Berry Curvature Map (Chern number: {Chern})')
plt.xlabel('kx (1/nm)')
plt.ylabel('ky (1/nm)')
plt.colorbar()
plt.show()


