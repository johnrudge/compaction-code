#! /usr/bin/python3

# ======================================================================
# mesh_gen.py
#
# Compute a mesh using gmsh
#
# John Rudge, University of Cambridge
# Laura Alisic, University of Cambridge
# Sander Rhebergen, University of Oxford
#
# Last modified: 26 Jan 2015 by Laura Alisic
# ======================================================================

from dolfin import *
from string import Template
import os
import numpy, sys, math

# ======================================================================

def cylinder_mesh_gen(filename, aspect, rel_radius, h, N):
    ''' Create a mesh with gmsh '''

    max_el_size = h / N
    geofile = "/tmp/" + filename + ".geo"
    mshfile = "/tmp/" + filename + ".msh"
    xmlfile = filename

    # gmsh code
    gmshtemplate = Template("""
    h         = $h;
    aspect    = $aspect;
    centrex   = h * aspect * 0.5;
    centrey   = h * 0.5;
    radius    = h * $rel_radius;
    cl1       = 1.0/5.0;
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
    Field[14] = Attractor;
    Field[14].NodesList = {5};
    Field[15] = Threshold;
    Field[15].IField = 14;
    Field[15].LcMin = cl1 / 40;
    Field[15].LcMax = cl1;
    Field[15].DistMin = 1.0*radius;
    Field[15].DistMax = 10.0*radius;
    Field[16] = Box;
    Field[16].VIn = cl1 / 10;
    Field[16].VOut = cl1;
    Field[16].XMin = centrex-1.0*radius;
    Field[16].XMax = centrex+1.0*radius;
    Field[16].YMin = centrey-1.0*radius;
    Field[16].YMax = centrey+1.0*radius;
    Field[17] = Min;
    Field[17].FieldsList = {15, 16};
    Background Field = 17;
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
    os.system("dolfin-convert "+ mshfile + " " + xmlfile)

    # Clean-up
    os.remove(geofile)
    os.remove(mshfile)

# ======================================================================

# EOF mesh_gen.py
