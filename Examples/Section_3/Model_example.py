import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from pybinding.repository import graphene



model = pb.Model(
    graphene.monolayer(),
    pb.translational_symmetry()      
                                                
                 )


x = model.system.x  # exctract the x positions 
y = model.system.y  # exctract the y positions 
subla = model.system.sublattices
hams = model.hamiltonian.todense # exctract the corresponding hamiltonian


model.plot()
print(x,y,subla,hams)
plt.show()