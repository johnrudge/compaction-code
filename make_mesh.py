#!/usr/bin/env python3

# ======================================================================
# Script make_mesh.py
#
# Script to create mesh *.xml file for later model run.
#
# Run using:
#     python3 make_mesh.py param_make_mesh.cfg
#
# where param_make_mesh.cfg is an example file with run parameters.
# One can provide the parameter file in the local run directory, and rename
# it at will.
#
# This script only runs in serial.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
# Sander Rhebergen, University of Oxford
# John Rudge, University of Cambridge
# Garth N. Wells <gnw20@cam.ac.uk>, University of Cambridge
# ======================================================================

# TODO: Allow for input of gmsh parameters to define refinement from
#       param_make_mesh.cfg file.

from mpi4py import MPI
from dolfinx.mesh import create_rectangle, DiagonalType
from dolfinx.io import XDMFFile
import numpy as np
import sys
import core
import mesh_gen
import mesh_gen_uniform
import mesh_gen_lessrefinement

# ======================================================================
# Run parameters
# ======================================================================

param_file = sys.argv[1]
param   = core.parse_param_file(param_file)

# Mesh parameters
read_mesh     = param['read_mesh']
cylinder_mesh = param['cylinder_mesh']
meshfile      = param['meshfile']
meshtype      = param['meshtype']
aspect        = param['aspect']
el            = param['el']
height        = param['height']
radius        = param['radius']

# ======================================================================
# Mesh
# ======================================================================

# Create mesh
print("**** Generating mesh . . . ")
if cylinder_mesh:
    # Create a mesh with gmsh
    #mesh_gen.cylinder_mesh_gen(filename=meshfile, \
    #mesh_gen_lessrefinement.cylinder_mesh_gen(filename=meshfile, \
    mesh_gen_uniform.cylinder_mesh_gen(filename=meshfile, \
                                aspect=aspect, \
                                N=el, \
                                h=height, \
                                rel_radius=(radius/height))
    with XDMFFile(MPI.COMM_WORLD, meshfile, "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")

else:
    if meshtype == "left/right":
        diagonal = DiagonalType.left_right
    elif meshtype == "left":
        diagonal = DiagonalType.left
    else:
        diagonal = DiagonalType.right
    mesh = create_rectangle(comm, \
                [np.array([0, 0]), np.array([aspect*height, height])], \
                [int(aspect*el), int(el)], diagonal=diagonal)
    with XDMFFile(MPI.COMM_WORLD, meshfile, "w") as xdmf:
       xdmf.write_mesh(mesh)

# EOF
