import pybinding as pb, matplotlib.pyplot as plt, numpy as np

pb.pltutils.use_style()

def QWZ_Model(t = 1, M = -1.5, a = 0.2, b = 1.5 * 0.2):

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

model = pb.Model(QWZ_Model(), pb.translational_symmetry())

solver = pb.solver.lapack(model)

scale = 2; steps = scale * 500; cap = scale * 2 * np.pi
kx = np.linspace(0, 2*cap, steps, endpoint=False)
ky = np.linspace(0, 2*cap, steps, endpoint=False)


Wavefunction_map = np.zeros((steps, steps, 2), dtype=complex)

for i in range(steps):
    for j in range (steps):
        k_point = [kx[i], ky[j]]
        k_space_function = solver.set_wave_vector(k = k_point )
        
        wavefunction_kx_ky = solver.eigenvectors

        Wavefunction_map[i,j] = wavefunction_kx_ky[:,0] # pick the band here 

    print(f"Wavefunction Progress: {i+1}/{steps}")


Wavefunction_map /= np.sum(Wavefunction_map)


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
berry_flux_total = 0.0 # Initial value of the integral
    
berry_curve_map = np.zeros((steps, steps))

for i in range(steps-1):
    for j in range(steps-1):

        # Energy band (n = 0 or 1)
        u = Wavefunction_map[i, j, :]  # wavefunction at (kx, ky)
        ux = Wavefunction_map[i+1, j, :]
        uxy = Wavefunction_map[i+1, j+1, :]
        uy = Wavefunction_map[i, j+1, :]

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

        chern = round(berry_flux_total / ((cap + cap ))) # have to round because numbers are not integers e.g. 0.99999 

    print(f"Berry Progress: {i+1}/{steps}")


plt.figure()
plt.contourf(kx, ky, berry_curve_map, cmap='plasma')
plt.title(f'Berry Curvature Map (Chern number: {chern})')
plt.xlabel('kx (1/nm)')
plt.ylabel('ky (1/nm)')
plt.colorbar()
plt.show()
