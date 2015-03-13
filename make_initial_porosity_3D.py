#!/usr/bin/env python

# ======================================================================
# Script make_initial_porosity_3D.py
#
# Script to create initial porosity field on a coarse mesh and large domain.
#
# Run using:
#     python make_initial_porosity_3D.py param_make_initial_porosity_3D.cfg
#
# where param_make_initial_porosity_3D.cfg determines the coarse mesh.
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
# Last modified: 21 Jan 2015 by Laura Alisic
# ======================================================================

from dolfin import *
import numpy, sys, math
import core
import physics_3D

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
meshfile      = param['meshfile']
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

if read_mesh:
    mesh = Mesh(meshfile)
else:
    mesh = BoxMesh(-1.55, -1.55, -0.05, \
           1.55, 1.55, 1.05, \
           int(3*el), int(3*el), int(el))
    #mesh = BoxMesh(-1.05, -1.05, -0.05, \
    #       1.05, 1.05, 1.05, \
    #       int(2*el), int(2*el), int(el))

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
phi_init = physics_3D.initial_porosity(param, Z)
phi0 = Function(Z)
phi0.interpolate(phi_init)

# Output initial porosity to HDF5 for later read-in
h5file_phi0 = HDF5File(comm, initial_porosity_out, "w")
h5file_phi0.write(phi0, "initial_porosity")
h5file_phi0.write(mesh, "large_mesh")
File("initial_porosity_original.pvd") << phi0

# Compute initial mean porosity
#mean_phi = assemble(phi0*dx)/(2.1*2.1*1.1)
mean_phi = assemble(phi0*dx)/(3.1*3.1*1.1)
info("**** Mean porosity = %g" % (mean_phi))

# EOF
