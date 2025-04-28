import pybinding as pb, matplotlib.pyplot as plt, numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene
from math import pi, sqrt

def wavy2(a, b): # This is thhe potential to be added 
    @pb.onsite_energy_modifier
    def potential(energy, x, y):
        v = np.sin(a * x)**2 + np.cos(b * y)**2
        return energy + v
    return potential

# circle with a hexagon cutout  inside of a triangle
hexagon = pb.regular_polygon(num_sides=6, radius=1.92, angle=np.pi/6)
circle = pb.circle(radius=4)
triangle = pb.regular_polygon(num_sides=3 ,radius=9, angle=np.pi/3)
rectangle = pb.rectangle(x=10, y=1)

shape = triangle - circle + hexagon + rectangle
shape.plot()


# here i fill the shape with monolayer graphene
plt.figure()
model1 = pb.Model(graphene.monolayer(),shape) 
model1.plot()

#this adds the potential to the confined system
model = pb.Model(graphene.monolayer(),shape,wavy2(a=0.6, b=0.9)) 
#model.shape.plot()

plt.figure()
model.plot() 
model.onsite_map.plot_contourf()
pb.pltutils.colorbar(label="U (eV)")

plt.figure() # from here I plot density of states 
solver = pb.solver.arpack(model, k=20)
ldos = solver.calc_spatial_ldos(energy=0.5, broadening=0.05)  
ldos.plot(site_radius=(0.03, 0.12))
pb.pltutils.colorbar(label="LDOS")

plt.show()
