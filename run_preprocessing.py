#!/usr/bin/env python3

# ======================================================================
# Script run_preprocessing.py
#
# Script to run a variety of preprocessing steps for model simulations.
#
# Run using:
#     python3 run_preprocessing.py
#
# This script only runs in serial.
#
# Adjust paths of scripts and parameter files as required.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
# Sander Rhebergen, University of Oxford
# John Rudge, University of Cambridge
# Garth N. Wells <gnw20@cam.ac.uk>, University of Cambridge
#
# Last modified: 13 Sept 2013 by Laura Alisic
# ======================================================================

import sys, os

# Create the mesh to do the numerical simulation on
cmd = "python3 make_mesh.py param_make_mesh.cfg"
os.system(cmd)

# Create the coarse initial random porosity field on a large mesh
cmd = "python3 make_initial_porosity.py param_make_initial_porosity.cfg"
os.system(cmd)

# Interpolate the coarse initial random porosity field onto the smaller
# mesh used in the simulation
cmd = "python3 interpolate_initial_porosity.py param_interpolate_initial_porosity.cfg"
os.system(cmd)

# EOF
