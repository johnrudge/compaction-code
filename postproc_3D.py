#!/usr/bin/env python3

# ======================================================================
# Script postproc_3D.py
#
# Does postprocessing for advection and compaction of a porous medium
# in 3-D. 
#
# No plotting is being done in this script.
#
# Run using:
#     python3 postproc_3D.py
#
# This code only works in serial.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
#
# Last modified: 23 Mar 2015 by Laura Alisic
# ======================================================================

from dolfin import *
import sys, math
import analysis
import numpy as np
import h5py
import os.path

# ======================================================================
# Run parameters
# ======================================================================

# FIXME:
# Hardcoded parameters for now
param             = {}
param['degree']   = 2
param['radius']   = 0.1
param['out_freq'] = 10
param['cfl']      = 0.5

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
#h5file_mesh = HDF5File(comm, "ln_porosity_0.h5", "r")
h5file_mesh = HDF5File(comm, "porosity_ln_0.h5", "r")
h5file_mesh.read(mesh, "mesh_file", False)
h5file_mesh.close()

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

#while (i == 0):
while 1:

    # Make sure file exists, otherwise stop the script
    #hdf_file = "ln_porosity_%d.h5" % (i)
    hdf_file = "porosity_ln_%d.h5" % (i)
    if os.path.exists(hdf_file):
        print('\n==== Output step ', i)
    else:
        print('\nLast output step reached, exiting.\n')
        break

    # Make sure vector fields in HDF5 are properly named; models run on
    # ARCHER have different names for some reason (different HDF version?)

    # Target name for datasets
    target_name = "/vector"

    # Find out what dataset is actually named for porosity
    #cmd = "h5ls -r ln_porosity_%d.h5/ln_porosity | grep 'vector..' | awk '{print $1}' > temp.out" % (i)
    cmd = "h5ls -r porosity_ln_%d.h5/porosity_ln | grep 'vector..' | awk '{print $1}' > temp.out" % (i)
    os.system(cmd)
    temp = open('temp.out', 'r')
    field_name = str.strip(temp.read())
    print('Porosity dataset name: ', field_name)
    temp.close()
    cmd = "rm temp.out"
    os.system(cmd)

    # If porosity dataset name isn't 'vector', rename to 'vector'
    if (field_name != target_name):
        print('Renaming porosity dataset...')
        #h5file_phi_temp = h5py.File("ln_porosity_%d.h5" % i, "r+")
        h5file_phi_temp = h5py.File("porosity_ln_%d.h5" % i, "r+")
        #h5file_phi_temp.move("ln_porosity" + field_name, "ln_porosity" + target_name)
        h5file_phi_temp.move("porosity_ln" + field_name, "porosity_ln" + target_name)
        h5file_phi_temp.close()
    else:
        print('No renaming required...')

    # Find out what dataset is actually named for compaction rate
    cmd = "h5ls -r compaction_rate_%d.h5/compaction_rate | grep 'vector..' | awk '{print $1}' > temp.out" % (i)
    os.system(cmd)
    temp = open('temp.out', 'r')
    field_name = str.strip(temp.read())
    print('Compaction rate dataset name: ', field_name)
    temp.close()
    cmd = "rm temp.out"
    os.system(cmd)

    # If compaction rate dataset name isn't 'vector', rename to 'vector'
    if (field_name != target_name):
        print('Renaming compaction rate dataset...')
        h5file_comp_temp = h5py.File("compaction_rate_%d.h5" % i, "r+")
        h5file_comp_temp.move("compaction_rate" + field_name, "compaction_rate" + target_name)
        h5file_comp_temp.close()
    else:
        print('No renaming required...')

    # Read HDF5 files from model simulation for further processing
    print('Reading data...')
    #h5file_phi  = HDF5File(comm, "ln_porosity_%d.h5" % i, "r")
    h5file_phi  = HDF5File(comm, "porosity_ln_%d.h5" % i, "r")
    h5file_comp = HDF5File(comm, "compaction_rate_%d.h5" % i, "r")

    #h5file_phi.read(phi, "ln_porosity")
    h5file_phi.read(phi, "porosity_ln")
    h5file_comp.read(comp, "compaction_rate")
    
    h5file_phi.close()
    h5file_comp.close()

    del h5file_phi
    del h5file_comp

    # Convert porosity from ln(phi) to phi
    phi_new = Expression(("exp(phi)"), phi = phi, element = Q.ufl_element())

    # Computation of integrals around cylinder
    analysis.cylinder_integrals_slice(phi_new, 'porosity', param, i)
    analysis.cylinder_integrals_slice(comp, 'compaction_rate', param, i)

    i += 1

# EOF postproc_3D.py
