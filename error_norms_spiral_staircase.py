#!/usr/bin/env python

# ======================================================================
# error_norms_spiral_staircase.py
#
# Computes error norms for numerical models with respect to analytical
# solutions of a 3-D torsion experiment using a spiral staircase initial
# porosity field.
#
# Only data points at the center plane z = 0.5 are used.
#
# Results are plotted vs resolution.
#
# Requires *.h5 output for the numerical and analytical solutions;
# fields are tailored to John's and Sander's codes.
#
# Run by using:
#     python error_norms_spiral_staircase.py
#
# Author:
# Laura Alisic, University of Cambridge
#
# Last modified: 1 June 2015 by Laura Alisic
# ======================================================================

from dolfin import *
import numpy, sys, math
import matplotlib as mpl
mpl.use('pdf')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.ioff()

# Set quadrature degree
ffc_parameters = dict(quadrature_degree=3, optimize=True)


# ======================================================================
# Functions
# ======================================================================

def compute_error(growth_an_plane, comp_num_plane):
    '''Compute error norm between numerical and analytical growth rate'''

    # Compute error norms
    info("**** Computing error norm on centerplane...")

    # Convert compaction rate to growth rate using factor: (1-phi0)/amplitude
    # = (1.0 - 0.05) / 5.0e-4 = 1.9e3
    growth_num_plane = project(comp_num_plane * 1.9e3)

    # Growth rate error
    an_norm  = norm(growth_an_plane)
    num_norm = norm(growth_num_plane)
    error    = (errornorm(growth_an_plane, growth_num_plane)) / an_norm

    info("---- Growth rate L2 norms: analytical %g, numerical %g, error %g\n" \
         % (an_norm, num_norm, error))

    return error

# ======================================================================
# Parameters
# ======================================================================

# The lists model_list and resolution must be the same length
 
model_list = ['new_N10_n5_amp5e-4_alpha28_R1_7_nexp1', \
              'new_N20_n5_amp5e-4_alpha28_R1_7_nexp1', \
              'new_N30_n5_amp5e-4_alpha28_R1_7_nexp1', \
              'new_N40_n5_amp5e-4_alpha28_R1_7_nexp1', \
              'new_N50_n5_amp5e-4_alpha28_R1_7_nexp1']

# Figure names
lin_fig_name = 'growth_rate_error_norms_linear_gridsize.pdf'
log_fig_name = 'growth_rate_error_norms_loglog_gridsize.pdf'

# List with inclusion radii corresponding to the above model pairs;  also 
# needs to be the same length as the model list vectors.
resolution = [10, 20, 30, 40, 50]

# Degree of vector function space
degree = 2

# Needed for interpolating fields without throwing an error
parameters['allow_extrapolation'] = True

# MPI command needed for HDF5
comm = mpi_comm_world()


# ======================================================================
# Loop over models
# ======================================================================

error_vals = []  # Growth rate error

for i, model in enumerate(model_list):
    print '\nModel:', model_list[i]
    print 'Resolution:', resolution[i]

    # Define files
    info("**** Defining input files...")

    # Input file from analytical code by John
    h5file_growth_an  = HDF5File(comm, ("../%s/analytical/output/growth_rate.h5" % (model_list[i])), "r")

    # Input file from numerical code by Sander
    h5file_comp_num = HDF5File(comm, ("../%s/numerical/compaction_rate_0.h5" % (model_list[i])), "r")

    # Full 3-D mesh and 2-D centerplane mesh
    info("**** Read in meshes...")
  
    # Read 3-D mesh from file
    mesh = Mesh()
    h5file_growth_an.read(mesh, "mesh_file", False)

    # Shift full 3-D mesh down by 0.5 in z such that plane mesh is in center
    for x in mesh.coordinates():
        x[2] -= 0.5
  
    # Read in centerplane mesh
    plane_mesh_file = ("circle2D_N%s.xml" % resolution[i])
    print plane_mesh_file
    plane_mesh = Mesh(plane_mesh_file)

    # Define function spaces on full mesh and on centerplane mesh
    info("**** Defining function spaces...")

    # Function spaces on full mesh
    Q = FunctionSpace(mesh, "Lagrange", degree-1)
    growth_an = Function(Q)
    comp_num  = Function(Q)

    # Function spaces on centerplane
    P = FunctionSpace(plane_mesh, "Lagrange", degree-1)
    growth_an_plane = Function(P)
    comp_num_plane = Function(P)

    # Read files
    info("**** Reading files...")
    h5file_growth_an.read(growth_an, "porosity")
    h5file_comp_num.read(comp_num, "compaction_rate")

    # Interpolate fields onto centerplane mesh
    info("**** Interpolating fields onto centerplane mesh...")
    growth_an_plane = interpolate(growth_an, P)
    comp_num_plane = interpolate(comp_num, P)

    # Compute errors
    error = compute_error(growth_an_plane, comp_num_plane)

    # Store error data in array
    error_vals.append(error)


print "Error array:", error_vals

# For direct plotting: 
#error_vals = [ 0.04351068644403, \
#               0.014067434270654355, \
#               0.011148660188023308, \
#               0.011187564503658102, \
#               0.011127906432382258 ]

# ======================================================================
# Plot results in linear space
# ======================================================================

info("\n**** Plotting results in linear space...")

# Start figure
lin_id       = 1
plt.figure(lin_id, figsize = (4,3))

# Plot data points
grid_size = [1.0/10.0, 1.0/20.0, 1.0/30.0, 1.0/40.0, 1.0/50.0]
plt.plot(grid_size, error_vals, marker = 'o', color = 'black')

# X axis
plt.xlim([0, 0.12])
plt.xlabel(r'Representative grid cell size')

# Y axis
lin_min = min(error_vals)
lin_max = max(error_vals)
plt.ylim(lin_min*0.80, lin_max*1.10)
plt.ylabel(r'$L_2$ error')

# Save figure to file
plt.savefig(lin_fig_name, bbox_inches='tight')


# ======================================================================
# Plot results in loglog space
# ======================================================================

info("**** Plotting results in loglog space...")

# Start figure
log_id       = 2
plt.figure(log_id, figsize = (4,3))

# Plot data points
plt.loglog(grid_size, error_vals, marker = 'o', color = 'black')

# X axis
plt.xlim([0.015, 0.12])
plt.xticks([0.02, 0.05, 0.10], \
           ['0.02', '0.05', '0.10'])
plt.xlabel(r'Representative grid cell size')

# Y axis
#plt.ylim([8e-3, 1.5e-1])
plt.ylim([1e-2, 1e-1])
plt.yticks([0.01, 0.02, 0.05, 0.1], \
           ['0.01', '0.02', '0.05', '0.10'])
plt.ylabel(r'$L_2$ error')

# Save figure to file
plt.savefig(log_fig_name, bbox_inches='tight')


info("\n\nDone!\n")

# EOF error_norms_spiral_staircase.py
