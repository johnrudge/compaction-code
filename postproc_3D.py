#!/usr/bin/env python

# ======================================================================
# Script postproc_3D.py
#
# Does postprocessing for advection and compaction of a porous medium
# in 3-D. 
#
# No plotting is being done in this script.
#
# Run using:
#     python postproc_3D.py
#
# This code only works in serial.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
#
# Last modified: 2 Feb 2015 by Laura Alisic
# ======================================================================

from dolfin import *
import sys, math
import analysis
import numpy as np

# ======================================================================
# Run parameters
# ======================================================================

# FIXME:
# Hardcoded parameters for now
param             = {}
param['degree']   = 2
param['radius']   = 0.1
param['out_freq'] = 10
param['cfl']      = 0.1

degree   = param['degree']
radius   = param['radius']
out_freq = param['out_freq']
cfl      = param['cfl']

# Needed for interpolating fields without throwing an error
parameters['allow_extrapolation'] = True

# MPI command needed for HDF5
comm = mpi_comm_world()

# ======================================================================
# Function spaces
# ======================================================================

# Read mesh from porosity file (can do from any of the .h5 files)
mesh = Mesh()
h5file_phi = HDF5File(comm, "porosity_0.h5", "r")
h5file_phi.read(mesh, "mesh_file", False)

# Define function space
Q = FunctionSpace(mesh, "Lagrange", degree-1)

# Define functions

# Porosity
phi = Function(Q)

# Compaction rate
comp = Function(Q)

# ======================================================================
# Post-processing in time loop 
# ======================================================================

# Loop over output steps
i = 0

while 1:

    print '==== Output step ', i

    try:
        # Input: HDF5 files from model simulation, for output step i
        h5file_phi  = HDF5File(comm, "porosity_%d.h5" % i, "r")
        h5file_comp = HDF5File(comm, "compaction_rate_%d.h5" % i, "r")

        h5file_phi.read(phi, "porosity")
        h5file_comp.read(comp, "compaction_rate")
    except:
        print 'Last output step reached.'
        break

    # Computation of integrals around cylinder
    analysis.cylinder_integrals_slice(phi, 'porosity', param, i)
    analysis.cylinder_integrals_slice(comp, 'compaction_rate', param, i)

    i += 1

# EOF postproc_3D.py
