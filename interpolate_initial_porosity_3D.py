#!/usr/bin/env python

# ======================================================================
# Script interpolate_initial_porosity_3D.py
#
# Script to read in mesh and interpolate larger porosity field onto.
# It produces a porosity h5 field that can be used for model simulations.
#
# Run using:
#     python interpolate_initial_porosity_3D.py param_interpolate_initial_porosity_3D.cfg
#
# This script only runs in serial.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
# Sander Rhebergen, University of Oxford
# John Rudge, University of Cambridge
# Garth N. Wells <gnw20@cam.ac.uk>, University of Cambridge
#
# Last modified: 29 Jan 2015 by Laura Alisic
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
cylinder_mesh = param['cylinder_mesh']
meshfile      = param['meshfile']
meshtype      = param['meshtype']
aspect        = param['aspect']
el            = param['el']
height        = param['height']
radius        = param['radius']

# Initial porosity parameters
initial_porosity_field = param['initial_porosity_field']
read_initial_porosity  = param['read_initial_porosity']
initial_porosity_in    = param['initial_porosity_in']
initial_porosity_out   = param['initial_porosity_out']
phiB      = param['phiB']
amplitude = param['amplitude']
angle_0   = math.pi * param['angle_0']
k_0       = math.pi * param['k_0']
nr_sines  = param['nr_sines']

# ======================================================================
# Mesh
# ======================================================================

# Read mesh from file; has to be exactly the same as used for the model
# computation!
info("**** Reading mesh file: %s", meshfile)
mesh = Mesh(meshfile)

# ======================================================================
# Function spaces
# ======================================================================

# Porosity
X = FunctionSpace(mesh, "DG", degree-1)

# ======================================================================
#  Initial porosity
# ======================================================================

# Set initial porosity field
info("**** Interpolating initial porosity field ...")

# MPI command needed for HDF5
comm = mpi_comm_world()

# Read in initial porosity
h5file_phi_in = HDF5File(comm, initial_porosity_in, "r")
large_mesh    = Mesh()
h5file_phi_in.read(large_mesh, "large_mesh", False)
P             = FunctionSpace(large_mesh, "Lagrange", 1)
phi_input     = Function(P)
h5file_phi_in.read(phi_input, "initial_porosity")

# Interpolate porosity onto final mesh
phi_proj = Function(X)
phi_proj.interpolate(phi_input)

# Output initial porosity to HDF5 for later read-in
h5file_phi_out = HDF5File(comm, initial_porosity_out, "w")
h5file_phi_out.write(phi_proj, "porosity")
h5file_phi_out.write(mesh, "mesh_file")
File("initial_porosity_interpolated.pvd") << phi_proj

# Compute initial mean porosity
if cylinder_mesh:
    mean_phi = assemble(phi_proj*dx)/(aspect*height*height - math.pi*radius**2)
else:
    mean_phi = assemble(phi_proj*dx)/(aspect*height*height)
info("**** Mean porosity = %g" % (mean_phi))

# EOF
