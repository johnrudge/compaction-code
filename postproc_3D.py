#!/usr/bin/env python

# ======================================================================
# Script postproc_3D.py
#
# Does postprocessing for advection and compaction of a porous medium
# in 3-D. Stripped down version; currently only computing compation rate
# integrals at step 0.
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
# Last modified: 16 April 2014 by Laura Alisic
# ======================================================================

from dolfin import *
import sys, math
import analysis
import numpy as np

# ======================================================================
# Functions
# ======================================================================

def determine_timestep(vel):
    """Figure out time step length"""

    mesh   = vel.function_space().mesh()

    W      = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    w      = TestFunction(W)

    volume = w.cell().volume
    L      = w*sqrt(dot(vel, vel))/volume*dx
    b      = assemble(L)
    umax   = b.norm("linf")

    h_min = mesh.hmin()
    dt     = cfl*h_min/max(1.0e-6, umax)
    #dt = 0.000413

    return dt

# ======================================================================
# Run parameters
# ======================================================================

# FIXME:
# Hardcoded parameters for now
param             = {}
param['degree']   = 2
param['radius']   = 0.2
param['out_freq'] = 10
param['cfl']      = 0.5

degree   = param['degree']
radius   = param['radius']
out_freq = param['out_freq']
cfl      = param['cfl']

# Input files
#h5file_phi    = HDF5File("porosity.h5", "r")
h5file_vel    = HDF5File("velocity.h5", "r")

# Needed for interpolating fields without throwing an error
parameters['allow_extrapolation'] = True

# ======================================================================
# Function spaces
# ======================================================================

# Read mesh from porosity file (can do from any of the .h5 files)
mesh = Mesh()
h5file_vel.read(mesh, "mesh_file")

# Define function spaces

# Velocity
V = VectorFunctionSpace(mesh, "Lagrange", degree)

# Pressure, porosity
#Q = FunctionSpace(mesh, "Lagrange", degree-1)

# Define functions
# Velocity
vel = Function(V)

# Porosity
#phi = Function(Q)

# ======================================================================
# Post-processing in time loop 
# ======================================================================

# Loop over output steps
nr_steps  = 0
strain    = []
time_step = []

# Loop left in place for later use, but only currently doing the 0-th timestep
#while 1:
while nr_steps == 0:
    i = nr_steps*out_freq

    try:
        # Read datasets for step i
        #h5file_phi.read(phi, "porosity_%d" % i)
        #h5file_phi.read(phi, "porosity")
        #h5file_vel.read(vel, "velocity_%d" % i)
        h5file_vel.read(vel, "velocity")
    except:
        print 'Last output step reached.'
        break

    # Determine time step length using velocity field
    dt = determine_timestep(vel)
    time_step.append(dt)         
    if nr_steps == 0: 
        strain = [0]
    else:
        strain.append(strain[nr_steps-1] + out_freq*time_step[nr_steps-1])

    print 'Step:', nr_steps, ', output step:', i, ', time step length:', time_step[nr_steps], \
          ', total strain:', strain[nr_steps]

    #if cylinder_mesh:
    # Computation of integrals around cylinder
    #analysis.cylinder_integrals_slice(phi, 'porosity', param, i)
    analysis.cylinder_integrals_slice(project(div(vel)), 'compaction_rate', param, i)

    nr_steps += 1

# EOF postproc_3D.py
