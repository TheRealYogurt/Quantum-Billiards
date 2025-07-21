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

shape = pb.circle(radius=1.5, center=(0, 0))
model =pb.Model(QWZ_Model(), shape)

# Eigenvalues and Eigenfunctions for the system in real space 
solver = pb.solver.lapack(model)
wavefunctions = solver.eigenvectors
eigenvalues = solver.eigenvalues
positions = solver.system.positions # real space positions of each atom - i think 
 
fscale = 6
r = np.column_stack(positions); steps = np.size(r[:,0])
kx = np.linspace(-fscale * 2*np.pi, fscale * 2*np.pi, steps)
ky = np.linspace(-fscale * 2*np.pi, fscale * 2*np.pi, steps)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")

mesh_step = kx[1]-kx[0]


n = 0; psi = wavefunctions[:, n] # eigen fucntion, index 0 being the fisrt
psi_reshaped = psi.reshape((len(r), 2))  # reshape 


M_xy = np.zeros((steps,steps),dtype=complex)

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

# The normalise Momentum distribution
M_xy /= np.sum(M_xy) 


