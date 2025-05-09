import pybinding as pb
import matplotlib.pyplot as plt
from math import sqrt

pb.pltutils.use_style()

def monolayer_1T_WTe2():
    a = 6.314769; b = 3.492485
    a1=[a, 0]; a2=[0, b]; t = 1
    lat = pb.Lattice( a1, a2) #a3 = [0, 0, c]) # Define rectangular lattice

    lat.add_sublattices(

        ('W1', [0.821873 * a, 0.750000 * b]), ('W2', [0.178127 * a, 0.250000 * b]),
        ('Te1',[0.081802 * a, 0.750000 * b]), ('Te2',[0.572013 * a, 0.250000 * b]),
        ('Te3',[0.427987 * a, 0.750000 * b]), ('Te4',[0.918198 * a, 0.250000 * b])
    )

    lat.add_hoppings(
        #main cell hoppings 
        ([0, 0], 'W1', 'Te3', t), ([0, 0], 'W1', 'Te2', t),
        ([0, 0], 'W2', 'Te3', t), ([0, 0], 'W2', 'Te2', t),
        ([0, 0], 'W2', 'Te1', t), ([0, 0], 'W1', 'Te4', t),

        #between neighbouring cells
        ([0,1],'Te1','W2', t),
        ([0,1],'Te3','W2', t),
        ([0,1],'W1','Te2', t),
        ([1,0],'W1','Te1', t),
        ([1,0],'Te4','W2', t),
        ([0,-1],'Te4','W1', t)
        )
    return lat

lattice = monolayer_1T_WTe2(); lattice.plot()

scale = 3; shape = pb.circle(radius = 3 * scale)
model = pb.Model(lattice,shape); #model.plot()
plt.show()
