import pybinding as pb, matplotlib.pyplot as plt, numpy as np
from scipy.interpolate import griddata

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

# Size:
scale = 5; x0 = 1.1 * scale; y0 = 0.5 * scale
shape = pb.Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])

# Parameters:
t = 1; M = 1.5; a = 0.2; b = 1.5 * 0.2

model = pb.Model(QWZ_Model(t=t, M=M, a=a, b=b), shape)

solver = pb.solver.lapack(model)
solver.solve()

# Extract eigenvalues and eigenvectors
eigenvalues = solver.eigenvalues
eigenvectors = solver.eigenvectors

# Choose eigenfunction - 0th is the lowest 
psi_occ = eigenvectors[:, 0]

# Convert to array of spinors per site
n_sites = model.system.num_sites
psi_site_spinors = psi_occ.reshape((n_sites, 2)) 

# Get positions
positions = np.column_stack(model.system.positions)

Hamiltonian = model.hamiltonian.todok()

# Match site index by approximate position
def find_index(pos_array, target, tol=1e-6):
    matches = np.all(np.isclose(pos_array[:, :2], target, atol=tol), axis=1)
    idx = np.where(matches)[0]
    if len(idx) == 0:
        raise IndexError(f"No site found at {target}")
    return idx[0]

berry_map = []

for x in np.arange(min(positions[:, 0]), max(positions[:, 0]) - a, a):
    for y in np.arange(min(positions[:, 1]), max(positions[:, 1]) - b, b):
        # Plaquette corners
        p0 = (x, y)
        p1 = (x + a, y)
        p2 = (x + a, y + b)
        p3 = (x, y + b)

        i0 = find_index(positions, p0)
        i1 = find_index(positions, p1)
        i2 = find_index(positions, p2)
        i3 = find_index(positions, p3)

        def sub_hamiltonian(i_from, i_to): # form 2x2 hamiltonian matrix

            H_sub = [[Hamiltonian[i_from, i_from], Hamiltonian[i_from, i_to]],
                     [Hamiltonian[i_to, i_from], Hamiltonian[i_to, i_to]]]

            return H_sub
        
        # spinors for each site 0 to 3
        psi_0 = psi_site_spinors[i0].reshape(2,1)
        psi_1 = psi_site_spinors[i1].reshape(2,1)
        psi_2 = psi_site_spinors[i2].reshape(2,1)
        psi_3 = psi_site_spinors[i3].reshape(2,1)

        # the expectation value of the spinors with and hamiltonian 
        def expectation_value(i_to, i_from, psi_from, psi_to):

            T = sub_hamiltonian(i_from = i_from, i_to = i_to)

            sub_phase = np.vdot(psi_from, T @ psi_to)

            return sub_phase
        
        U01 = expectation_value(i1, i0, psi_0, psi_1)
        U12 = expectation_value(i2, i1, psi_1, psi_2)
        U23 = expectation_value(i3, i2, psi_2, psi_3)
        U30 = expectation_value(i0, i3, psi_3, psi_0)

        U = U01 + U12 + U23 + U30 # exp(φ)

        berry_phase = np.angle(U) # φ

        berry_curvature = berry_phase / (a * b) # phase / area of the shape

        berry_map.append(((x + a / 2, y + b / 2), berry_curvature)) # curvature at point x/2,y/2

# Extract x, y, and curvature values from berry_map
xy = np.array([pt for pt, _ in berry_map])  # x and y coordinates
F = np.array([f for _, f in berry_map])  # Berry curvature values

# Create meshgrid for contour plot
x_vals = xy[:, 0]; y_vals = xy[:, 1]

# Create grid for contour plot
xi = np.linspace(np.min(x_vals), np.max(x_vals), 20000)  
yi = np.linspace(np.min(y_vals), np.max(y_vals), 20000)
X, Y = np.meshgrid(xi, yi)

# Interpolate Berry curvature
F_grid = griddata((x_vals, y_vals), F, (X, Y), method='cubic')

# plot
plt.figure(figsize=(8, 6))
cp = plt.contourf(X, Y, F_grid, cmap='plasma')  # Use levels for better control of contour levels
plt.colorbar(cp, label='Berry curvature')
plt.title("Berry Curvature Contour Plot")
plt.xlabel("x"); plt.ylabel("y")
plt.axis('equal'); plt.tight_layout()
plt.show()
