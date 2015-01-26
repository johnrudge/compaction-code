#!/usr/bin/env python

# ======================================================================
# Script make_initial_porosity.py
#
# Script to create initial porosity field on a coarse mesh and large domain.
#
# Run using:
#     python make_initial_porosity.py param_make_initial_porosity.cfg
#
# where param_make_initial_porosity.cfg determines the coarse mesh.
# One can provide the parameter file in the local run directory, and rename
# it at will.
#
# The created porosity field can then later be interpolated onto a mesh
# that is used during the model simulation.
#
# This script only runs in serial.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
# Sander Rhebergen, University of Oxford
# John Rudge, University of Cambridge
# Garth N. Wells <gnw20@cam.ac.uk>, University of Cambridge
#
# Last modified: 26 Jan 2015 by Laura Alisic
# ======================================================================

from dolfin import *
import numpy, sys, math
import core
import physics

# Needed for interpolating fields without throwing an error
parameters['allow_extrapolation'] = True

# ======================================================================
# Run parameters
# ======================================================================

param_file = sys.argv[1]
param   = core.parse_param_file(param_file)

# FEM parameters
degree   = param['degree']

# Mesh parameters
read_mesh     = param['read_mesh']
meshtype      = param['meshtype']
aspect        = param['aspect']
el            = param['el']
height        = param['height']

# Initial porosity parameters
initial_porosity_field = param['initial_porosity_field']
initial_porosity_out   = param['initial_porosity_out']
phiB      = param['phiB']
amplitude = param['amplitude']
angle_0   = math.pi * param['angle_0']
k_0       = math.pi * param['k_0']
nr_sines  = param['nr_sines']

# MPI command needed for HDF5
comm = mpi_comm_world()

# ======================================================================
# Mesh
# ======================================================================

# Create mesh on large square without inclusion.
info("**** Generating mesh . . . ")
mesh = RectangleMesh(0, 0, aspect*height, height, \
                     int(aspect*el), int(el), meshtype)

# Smallest element size. Used to determine time step
h_min = MPI.min(comm, mesh.hmin())
h_max = MPI.max(comm, mesh.hmax())

# Minimum and maximum element size
info("hmin = %g, hmax = %g" % (h_min, h_max))

# Shift mesh such that the center is at the origin
print "Shifting mesh"
for x in mesh.coordinates():
    x[0] -= 0.5*height*aspect
    x[1] -= 0.5*height

    # Shift elements at side boundaries to avoid roundoff errors and
    # ensuing problems with periodic boundary mapping when mesh is
    # created with gmsh

    # Distance away from boundary for this to be applied
    margin    = 1.0e-4

    # The larger this number, the more digits are included
    precision = 1.0e6
    if (x[0] < 0.5*height*aspect+margin) or (x[0] > 0.5*height*aspect-margin):
        if x[0] > 0:
            x[0] = int(x[0] * precision + 0.5) / precision
        else:
            x[0] = int(x[0] * precision - 0.5) / precision
        if x[1] > 0:
            x[1] = int(x[1] * precision + 0.5) / precision
        else:
            x[1] = int(x[1] * precision - 0.5) / precision

print "End mesh shift"

# ======================================================================
# Function spaces
# ======================================================================

# Porosity
Z = FunctionSpace(mesh, "Lagrange", degree-1)

# ======================================================================
#  Initial porosity
# ======================================================================

# Set initial porosity field
info("**** Defining initial porosity field ...")

# Interpolate initial porosity
phi_init = physics.initial_porosity(param, Z)
phi0 = Function(Z)
phi0.interpolate(phi_init)

# Output initial porosity to HDF5 for later read-in
h5file_phi0 = HDF5File(comm, initial_porosity_out, "w")
h5file_phi0.write(phi0, "initial_porosity")
h5file_phi0.write(mesh, "large_mesh")
File("initial_porosity_original.pvd") << phi0

# Compute initial mean porosity
mean_phi = assemble(phi0*dx)/(aspect*height*height)
info("**** Mean porosity = %g" % (mean_phi))

# EOF
