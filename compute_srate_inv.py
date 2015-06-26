#!/usr/bin/env python

# ======================================================================
# compute_srate_inv.py
#
# Computes second invariant of the strain-rate.
#
# Requires *.h5 output file form compaction-torsion code.
#
# Run by using:
#     python compute_srate_inv.py [porosity file]
#
# Author:
# Laura Alisic, University of Cambridge
#
# Last modified: 26 June 2015 by Laura Alisic
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
h5file_us = HDF5File(comm, infile, "r")

# Read mesh from file
info("Read mesh")
mesh = Mesh()
h5file_us.read(mesh, "mesh_file", False)

# Define function space
info("Define function space and functions")
V = VectorFunctionSpace(mesh, "Lagrange", degree)
Q = FunctionSpace(mesh, "Lagrange", degree-1)

# Define functions
us = Function(V)
srate_inv = Function(Q)

# Read file
info("Read file")
h5file_us.read(us, "velocity")

# Compute strain rate field
info("Compute strain rate")
Id = Identity(us.cell().geometric_dimension())
srate = sym(grad(us)) - Id * div(us) / 3.0
srate_inv = sqrt(0.5*inner(srate, srate))

# Output
info("Output strain rate")
srate_out = project(srate_inv, Q)
srate_out.rename("srate_inv", "")
srate_file = File("srate_inv_0.xdmf")
srate_file << srate_out

# EOF compute_srate_inv.py
