#!/usr/bin/env python

# ======================================================================
# Script interpolate_initial_porosity.py
#
# Script to read in mesh and interpolate larger porosity field onto.
# It produces a porosity h5 field that can be used for model simulations.
#
# Run using:
#     python interpolate_initial_porosity.py param_interpolate_initial_porosity.cfg
#
# This script only runs in serial.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
# Sander Rhebergen, University of Oxford
# John Rudge, University of Cambridge
# Garth N. Wells <gnw20@cam.ac.uk>, University of Cambridge
#
# Last modified: 13 Sept 2013 by Laura Alisic
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

# Minimum and maximum element size
h_min = MPI.min(mesh.hmin())
h_max = MPI.max(mesh.hmax())
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

# Subdomain for periodic boundary condition (left and right boundaries)
class PeriodicBoundary(SubDomain):
    """Define periodic boundaries"""
    def __init__(self, tol):
        SubDomain.__init__(self, tol)

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],-0.5*height*aspect, 1.0e-11)

    def map(self, x, y):
        """Map slave entity to master entity"""
        y[0] = x[0] - height*aspect
        y[1] = x[1]

# Create an object to prevect director going out of scope. Might fix
# later.
pbc = PeriodicBoundary(1.0e-6)

# ======================================================================
# Function spaces
# ======================================================================

# Porosity
X = FunctionSpace(mesh, "Lagrange", degree-1, constrained_domain=pbc)

# ======================================================================
#  Initial porosity
# ======================================================================

# Set initial porosity field
info("**** Interpolating initial porosity field ...")

# Read in initial porosity
h5file_phi_in = HDF5File(initial_porosity_in, "r")
large_mesh    = Mesh()
h5file_phi_in.read(large_mesh, "large_mesh")
P             = FunctionSpace(large_mesh, "Lagrange", 1)
phi_input     = Function(P)
h5file_phi_in.read(phi_input, "initial_porosity")

# Interpolate porosity onto final mesh
phi_proj = Function(X)
phi_proj.interpolate(phi_input)

# Output initial porosity to HDF5 for later read-in
h5file_phi_out = HDF5File(initial_porosity_out, "w")
h5file_phi_out.write(phi_proj, "initial_porosity")
h5file_phi_out.write(mesh, "mesh_file")
File("initial_porosity_interpolated.pvd") << phi_proj

# Compute initial mean porosity
if cylinder_mesh:
    mean_phi = assemble(phi_proj*dx)/(aspect*height*height - math.pi*radius**2)
else:
    mean_phi = assemble(phi_proj*dx)/(aspect*height*height)
info("**** Mean porosity = %g" % (mean_phi))

# EOF
