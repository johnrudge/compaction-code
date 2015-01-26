#!/usr/bin/env python

# ======================================================================
# Script make_mesh.py
#
# Script to create mesh *.xml file for later model run.
#
# Run using:
#     python make_mesh.py param_make_mesh.cfg
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
#
# Last modified: 26 Jan 2015 by Laura Alisic
# ======================================================================

# TODO: Allow for input of gmsh parameters to define refinement from
#       param_make_mesh.cfg file.

from dolfin import *
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

# Buid mesh using CSG via CGAL (FIXME: Not periodic)
#r = Rectangle(0.0, 0.0, height*aspect, height*aspect)
#c = Circle (0.5*height*aspect, 0.5* height*aspect, radius)
#g2d = r - c
#mesh = Mesh(g2d, 150)

# Create mesh
info("**** Generating mesh . . . ")
if cylinder_mesh:

    #if not has_cgal():
    #    info("DOLFIN must be compiled with CGAL to run the meshing.")
    #    sys.exit()
    #outside  = Rectangle(0, 0, aspect*height, height)
    #inside   = Circle(0.5*aspect*height, 0.5*height, radius, int(0.5*el))
    #geometry = outside - inside
    #mesh     = Mesh(geometry, 100)

    # Create a mesh with gmsh
    #mesh_gen_uniform.cylinder_mesh_gen(filename=meshfile, \
    #mesh_gen.cylinder_mesh_gen(filename=meshfile, \
    mesh_gen_lessrefinement.cylinder_mesh_gen(filename=meshfile, \
                                aspect=aspect, \
                                N=el, \
                                h=height, \
                                rel_radius=(radius/height))
    mesh = Mesh(meshfile)
else:
    mesh = RectangleMesh(0, 0, aspect*height, height, \
                         int(aspect*el), int(el), meshtype)
    mesh_out = File(meshfile)
    mesh_out << mesh

# EOF
