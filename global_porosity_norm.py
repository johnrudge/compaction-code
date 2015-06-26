#!/usr/bin/env python

# ======================================================================
# global_porosity_norm.py
#
# Computes global porosity norm as \int | phi - phi_0 | ^2 dx
#
# Requires *.h5 output file form compaction-torsion code.
#
# Run by using:
#     python global_porosity_norm.py [porosity file]
#
# Author:
# Laura Alisic, University of Cambridge
#
# Last modified: 25 June 2015 by Laura Alisic
# ======================================================================

from dolfin import *
import numpy, sys, math
import matplotlib as mpl
mpl.use('pdf')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.ioff()

# Set quadrature degree
ffc_parameters = dict(quadrature_degree=3, optimize=True)

# Degree of vector function space
degree = 2

# Needed for interpolating fields without throwing an error
parameters['allow_extrapolation'] = True

# MPI command needed for HDF5
comm = mpi_comm_world()

# Define input file
infile = sys.argv[1]
h5file_phi = HDF5File(comm, infile, "r")

# Read mesh from file
info("Read mesh")
mesh = Mesh()
h5file_phi.read(mesh, "mesh_file", False)

# Define function space
info("Define function space and functions")
Q = FunctionSpace(mesh, "Lagrange", degree-1)

# Define functions
phi = Function(Q)
phi_ln = Function(Q)
phi0 = Function(Q)

# Read file
info("Read file")
h5file_phi.read(phi_ln, "porosity_ln")
#h5file_phi.read(phi_ln, "ln_porosity")

# Compute porosity field
info("Compute porosity field")
phi = project(exp(phi_ln), Q)
 
# Porosity variation
info("Compute porosity variation")
phi0_val = 0.05
phi0 = project(phi0_val, Q)
phi_var = project(phi - phi0, Q)

# Mean porosity
info("Compute mean porosity")
height = 1.0
cylinder_radius = 1.0
inclusion_radius = 0.1
volume = math.pi*height*cylinder_radius**2 - math.pi*inclusion_radius**2
phi_mean = assemble(phi*dx) / volume

# Global porosity norms
info("Compute porosity norms")
phi_norm = norm(phi)
phi_var_norm = norm(phi_var)
#phi_error_norm = errornorm(phi, phi0)

# Output
info("\nGlobal phi mean: %g" % (phi_mean))
info("Global phi norm: %g" % (phi_norm))
info("Global (phi-phi0) norm: %g\n" % (phi_var_norm))
#info("Global (phi-phi0) error norm: %g\n" % (phi_error_norm))

# EOF global_porosity_norm.py
