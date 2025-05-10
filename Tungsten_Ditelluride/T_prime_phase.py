import pybinding as pb
import matplotlib.pyplot as plt
from math import sqrt

pb.pltutils.use_style()

def monolayer_1T_WTe2():
    a = 6.314769; b = 3.492485; a1=[a, 0]; a2=[0, b] # lattice vectors

    # Onsite energies p -> Te , d -> W, lattice hopping energies 
    mu_d = 0.22665; mu_p = -1.66528; t = [0.95946, -0.25719, 0.37810, -0.15573]  

    lat = pb.Lattice( a1, a2) # Rectangular 2D lattice

    lat.add_sublattices(
        #main cell
        ('W1', [0.821873 * a, 0.750000 * b], mu_d), ('W2', [0.178127 * a, 0.250000 * b], mu_d), 
        ('Te1',[0.081802 * a, 0.750000 * b], mu_p), ('Te2',[0.572013 * a, 0.250000 * b], mu_p),
        ('Te3',[0.427987 * a, 0.750000 * b], mu_p), ('Te4',[0.918198 * a, 0.250000 * b], mu_p)
    )

    lat.add_hoppings(
        #main cell hoppings 
        ([0, 0], 'W1', 'Te3', t[1]), 
        ([0, 0], 'W1', 'Te2', t[0]), 
        ([0, 0], 'W2', 'Te3', t[0]), 
        ([0, 0], 'W2', 'Te2', t[1]),
        ([0, 0], 'W2', 'Te1', t[2]), 
        ([0, 0], 'W1', 'Te4', t[3]),

        #between neighbouring cells
        ([0,1],'Te1','W2', t[2]),
        ([0,1],'Te3','W2', t[3]),
        ([0,1],'W1','Te2', t[0]),
        ([1,0],'W1','Te1', t[3]),
        ([1,0],'Te4','W2', t[3]),
        ([0,-1],'Te4','W1', t[2])
        )
    return lat

lattice = monolayer_1T_WTe2(); lattice.plot()

scale = 3; shape = pb.circle(radius = 3 * scale)
model = pb.Model(lattice,shape); #model.plot()
plt.show()
