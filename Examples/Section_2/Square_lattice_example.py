import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


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

# we can quickly set a shorter unit length `d`
lattice = square_lattice(d, t)
lattice.plot()
plt.show()