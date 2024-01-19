#!/usr/bin/env python3

# ======================================================================
# mesh_gen.py
#
# Compute a mesh using gmsh
#
# John Rudge, University of Cambridge
# Laura Alisic, University of Cambridge
#
# ======================================================================

from string import Template
import os
import numpy, sys, math
import meshio

# ======================================================================

def cylinder_mesh_gen(filename, aspect, rel_radius, h, N):
    ''' Create a mesh with gmsh '''

    max_el_size = h / N
    geofile = "/tmp/" + filename + ".geo"
    mshfile = "/tmp/" + filename + ".msh"
    xdmffile = filename

    # gmsh code
    gmshtemplate = Template("""
    h         = $h;
    aspect    = $aspect;
    centrex   = h * aspect * 0.5;
    centrey   = h * 0.5;
    radius    = h * $rel_radius;
    cl1       = 1.0;
    Point(1)  = {0.0, 0.0, 0.0, cl1};
    Point(2)  = {h*aspect, 0.0, 0.0, cl1};
    Point(3)  = {h*aspect, h, 0.0, cl1};
    Point(4)  = {0.0, h, 0.0, cl1};
    Point(5)  = {centrex, centrey, 0.0, cl1};
    Point(6)  = {centrex, centrey+radius, 0.0, cl1};
    Point(7)  = {centrex, centrey-radius, 0.0, cl1};
    Point(8)  = {centrex+radius, centrey, -0.0, cl1};
    Point(9)  = {centrex-radius, centrey, 0.0, cl1};
    Line(1)   = {3, 2};
    Line(2)   = {2, 1};
    Line(3)   = {1, 4};
    Line(4)   = {4, 3};
    Circle(5) = {9, 5, 6};
    Circle(6) = {6, 5, 8};
    Circle(7) = {8, 5, 7};
    Circle(8) = {7, 5, 9};
    Line Loop(14) = {4, 1, 2, 3, -8, -7, -6, -5};
    Plane Surface(14) = {14};
    """)
    gmshcode = gmshtemplate.substitute(rel_radius=rel_radius, h=h, \
                                           aspect=aspect)

    # Write gmsh geo file
    f = open(geofile, 'w')
    f.write(gmshcode)
    f.close()

    # File conversion
    os.system("gmsh " + geofile + " -2 -clmax " + str(max_el_size))
    msh = meshio.read(mshfile)
    triangles = meshio.Mesh(points=msh.points, cells={"triangle": msh.get_cells_type("triangle")})
    meshio.write(xdmffile, triangles)

    # Clean-up
    os.remove(geofile)
    os.remove(mshfile)

# ======================================================================

# EOF mesh_gen.py
