#!/usr/bin/env python

# ======================================================================
# Script make_initial_porosity_spiral_staircase.py
#
# Script to create initial porosity field for the spiral staircas
# torsion benchmark.
#
# Run using:
#     python make_initial_porosity_spiral_staircase.py param_spiral_staircase.cfg
#
# where param_spiral_staircase.cfg provides needed parameters. 
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
# John Rudge, University of Cambridge
#
# Last modified: 12 Feb 2015 by Laura Alisic
# ======================================================================

# Import dolfin
from dolfin import *

# Import general python modules
import sys
import math
import numpy as np
import scipy.special as sp

# Import own modules
import core

# Needed for interpolating fields without throwing an error
parameters['allow_extrapolation'] = True

# ======================================================================
# Run parameters
# ======================================================================

# Get parameters from parameter file
param_file   = sys.argv[1]
param        = core.parse_param_file(param_file)

# FEM parameters
degree       = param['degree']

# Mesh parameters
mesh_file    = param['mesh_file']
height       = param['height']
radius       = param['radius']

# Initial porosity parameters
porosity_out = param['porosity_out']
phi0         = param['phi0']
amplitude    = param['amplitude']
angle0       = math.pi * param['angle0']
n            = param['n']

# MPI command needed for HDF5
comm = mpi_comm_world()

# ======================================================================
# Mesh
# ======================================================================

# Reading in mesh
info("\n**** Reading mesh ... ")
mesh = Mesh(mesh_file)

# ======================================================================
# Function spaces
# ======================================================================

# Porosity
Z = FunctionSpace(mesh, "Lagrange", degree-1)

# ======================================================================
# Initial porosity
# ======================================================================

info("**** Computing initial porosity field ...")

# Compute initial porosity field
class PorosityField(Expression):
    def eval(self, value, x):

        # Cylindrical coordinates: radius
        rho = sqrt(x[0]*x[0] + x[1]*x[1])

        # Cylindrical coordinates: azimuth
        psi = math.atan2(x[1], x[0])
        
        # Cylindrical coordinates: height
        z = x[2] - 0.5

        # Initial angle
        a = 1.0 / (radius * tan(angle0))

        # Derivative of Bessel function of the first kind, n-th order, p-th zero
        # Using p = 1 since this is the fastest growing
        jnp = float(sp.jnp_zeros(n, 1))        

        # Bessel functions of the first kind
        jn1 = sp.jv(n, jnp * rho) 
        jn2 = sp.jv(n, jnp)

        # Porosity field
        value[0] = phi0 + amplitude * (jn1 / jn2) * cos(n * (psi + a * z))

# Interpolate initial porosity
phi_func = PorosityField()
phi_init = Function(Z)
phi_init.interpolate(phi_func)

# ======================================================================
# Output 
# ======================================================================

# Output initial porosity to HDF5 for later read-in
info("**** Writing initial porosity field ...")
h5file_phi = HDF5File(comm, porosity_out, "w")
h5file_phi.write(phi_init, "porosity")
h5file_phi.write(mesh, "mesh_file")

# Output initial porosity field to PVD for checking purposes
File("initial_porosity_staircase.pvd") << phi_init

info("\nDone!\n")

# EOF make_initial_porosity_spiral_staircase.py
