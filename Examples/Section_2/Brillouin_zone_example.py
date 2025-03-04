import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def monolayer_graphene():
    a = 0.24595   # [nm] unit cell length 0.24595
    a_cc = 0.142  # [nm] carbon-carbon distance 0.142
    t = -2.8      # [eV] nearest neighbour hopping -2.8

    lat = pb.Lattice(a1=[a, 0],
                     a2=[a/2, a/2 * sqrt(3)])
    

    lat.add_sublattices(('A', [0, -a_cc/2]),
                        ('B', [0,  a_cc/2])) 
    lat.add_hoppings(
        # inside the main cell
        ([0,  0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )
    return lat


lattice = monolayer_graphene()
lattice.plot_brillouin_zone()





def square_lattice(d, t):
    lat = pb.Lattice(a1=[d, 2], a2=[3, d])
    lat.add_sublattices(('A', [5, 5]))
    lat.add_hoppings(([0, p], 'A', 'A', t),
                     ([p, 0], 'A', 'A', t),
                     ([p/2, 0], 'A', 'A', t),
                     ([0, p/2], 'A', 'A', t))
    return lat

p = 2.5
t = 1
d = 1

lattice = square_lattice(d, t)
lattice.plot_brillouin_zone()




plt.show()