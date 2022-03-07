#!/usr/bin/env python3

# ======================================================================
# Script plot_regimes.py
#
# Does postprocessing for advection and compaction of a porous medium.
#
# Run using:
#     plot_regimes.py <data>.txt
#
# where <data>.txt is a file that contains a suite of model results in 
# the following columns:
#
# <alpha>  <r_zeta>  <max strain>  <shearbanding 0/1>
#
# This code only works in serial.
#
# Author:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
#
# Last modified: 07 Apr 2015 by Laura Alisic
# ======================================================================

import sys, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import griddata

plt.ioff()

# ======================================================================
# Functions
# ======================================================================

# Read data from <data>.txt
def read_data(data_file):
    """Read in data from summary file, store in numpy arrays"""

    file_in = open(data_file,"r")

    # Arrays to store data in for further plotting etc
    a_vals = [] # alpha value
    r_vals = [] # r_zeta value
    t_vals = [] # max strain/time reached
    s_vals = [] # shearbanding flag: 0 is no shear bands; 1 shear bands

    line_nr = 1
    for line in file_in.readlines():
        a, r, t, s = line.split()

        a_vals.append(float(a))
        #r_vals.append(math.log10(float(r)))
        r_vals.append(float(r))
        t_vals.append(float(t))
        s_vals.append(int(s))

    file.close(file_in)

    return a_vals, r_vals, t_vals, s_vals

# Read data from lobe intersections file
def read_intersections(intersections_file):
    """Read in data from lobe intersections file, store in numpy arrays"""

    file_in = open(intersections_file,"r")

    # Arrays to store data in for further plotting etc
    a_vals = [] # angles at which porosity intersects background phi0

    line_nr = 1
    for line in file_in.readlines():
        a = line.rstrip(' \n')
        a_vals.append(float(a))

    file.close(file_in)
    
    return a_vals

# Compute width of positive porosity lobes
def compute_lobe_width(int_vals):
    """Compute average width of positive porosity lobes"""

    # Compute widths of all lobes
    if len(int_vals) == 4:
        width_pos1 = int_vals[1] - int_vals[0]
        width_neg1 = int_vals[2] - int_vals[1]
        width_pos2 = int_vals[3] - int_vals[2]
        # The last negative lobe will be across 2pi
        width_neg2 = int_vals[0] + (2.0*math.pi - int_vals[3])

    # Too many crossings: positive lobes will be the widest intervals
    # (only an issue for random models)
    # FIXME: Check this!!
    elif len(int_vals) > 4:

        # Compute lobe widths
        width_vals = [] 
        for j in range(len(int_vals)-1):
            width = int_vals[j+1] - int_vals[j]
            width_vals.append(width)
        width = int_vals[0] + (2.0*math.pi - int_vals[-1])
        width_vals.append(width)

        # Largest ones are positive
        width_vals.sort()
        width_pos1 = width_vals[-1]
        width_pos2 = width_vals[-2]

    # Fewer crossings: what to do?
    # FIXME: Check this!!
    else:
        width_neg = int_vals[1] - int_vals[0]
        width_pos1 = math.pi - width_neg       
        width_pos2 = width_pos1

    # Compute average of positive lobes
    ave_pos = 0.5*(width_pos1 + width_pos2)

    # Scale average by 0.5pi
    # Value > 1: wider lobe than standard (shear banding)
    # Value = 1: lobe exactly 90 degrees wide
    # Value < 1: narrower lobe than standard (advected lobes)
    ave_pos_scaled = ave_pos / (0.5*math.pi)

    return ave_pos_scaled

# ======================================================================
# Run parameters
# ======================================================================

data_file       = sys.argv[1]
cylinder_flag   = 0
random_flag     = 1
analytical_flag = 0
#fig_name        = 'random_inclusion_regimes_simple.pdf'
#fig_name        = 'random_noinclusion_regimes_simple.pdf'
fig_name        = 'random_regimes_simple.pdf'

# ======================================================================
# Preparation for figure
# ======================================================================

# Read summary data
[a_vals, r_vals, t_vals, s_vals] = read_data(data_file)

# Data points; read in lobe intersections if cylinder model
points             = []
width_vals         = []
r_vals_shear       = []
r_vals_noshear     = []
r_vals_some_shear  = []
a_vals_shear       = []
a_vals_noshear     = []
a_vals_some_shear  = []
compaction_anomaly = []
shear_anomaly      = []
for i in range(len(t_vals)):
    points.append([r_vals[i], a_vals[i]])

    # Cylinder models: compute lobe widths around the cylinder at max strain
    if cylinder_flag:

        # Read lobe intersections file
        intersections_file = '../alpha%d_rzeta%g/lobe_intersections_max_strain.txt' \
                             % (a_vals[i], r_vals[i])
        #                     % (a_vals[i], 10**r_vals[i])
        #intersections_file = '../alpha%d_rzeta%g_amplitude3e-2/lobe_intersections_max_strain_cleaned.txt' \
        #                     % (a_vals[i], r_vals[i])
        #intersections_file = '../alpha%d_rzeta%g_amplitude5e-2_phi0_0.1/lobe_intersections_max_strain.txt' \
        #                     % (a_vals[i], r_vals[i])
        int_vals = read_intersections(intersections_file) 

        # From intersections, compute average width of positive lobes
        width = compute_lobe_width(int_vals)
        width_vals.append(width)

    # Random models: separate points according to yes/no shear banding
    if random_flag:

        # If no shear, add points to noshear data vectors
        if s_vals[i] == 0:
            r_vals_noshear.append(r_vals[i])
            a_vals_noshear.append(a_vals[i])

        # If shear banding, add points to shear data vectors
        if s_vals[i] == 1:
            r_vals_shear.append(r_vals[i])
            a_vals_shear.append(a_vals[i])

        # If shear banding away from inclusion, add points to some_shear data vectors
        if s_vals[i] == 2:
            r_vals_some_shear.append(r_vals[i])
            a_vals_some_shear.append(a_vals[i])

    # Analytical: compute analytical porosity anomalies given max strain and parameters
    if analytical_flag:

        gamma = t_vals[i]
        B     = 1.0 / (r_vals[i] + (4.0/3.0))

        # Compute anomaly for compaction: only meaningful for cylinder models
        comp_val = (4.0 * B / (1.0 + B)) * math.sin(gamma / 2.0) 
        compaction_anomaly.append(comp_val)
         
        # Compute anomaly for shear banding
        amplitude = 0.03
        phi0      = 0.05
        f_gamma   = 1.0 + 0.5*gamma * (gamma + math.sqrt(4.0 + gamma**2))
        shear_val = amplitude * (f_gamma**(a_vals[i] * B * (1.0 - phi0)))
        shear_anomaly.append(shear_val)

# Grid to interpolate data onto
#r_min = math.log10(1.0)
#r_max = math.log10(100.0)
#r_int = 0.1j
r_min = 0.0
r_max = 20.0
r_int = 2j
a_min = 0
a_max = 50
a_int = 5j

grid_x, grid_y = np.mgrid[r_min:r_max:r_int, a_min:a_max:a_int]

# Reshape 1-D arrays to 2-D
r_array = np.array(r_vals)
a_array = np.array(a_vals)
t_array = np.array(t_vals)
if cylinder_flag:
    width_array      = np.array(width_vals)
if analytical_flag:
    compaction_array = np.array(compaction_anomaly)
    shear_array      = np.array(shear_anomaly)

ncols = 4
r_array.shape = (r_array.size//ncols, ncols)
a_array.shape = (a_array.size//ncols, ncols)
t_array.shape = (t_array.size//ncols, ncols)
if cylinder_flag:
    width_array.shape = (width_array.size//ncols, ncols)
if analytical_flag:
    compaction_array.shape = (compaction_array.size//ncols, ncols)
    shear_array.shape = (shear_array.size//ncols, ncols)
   
# ======================================================================
# Create figure
# ======================================================================

# Start figure
#plt.figure(1, figsize = (5.5,4)) # if no legend on the side
#plt.figure(1, figsize = (7,4))  # if legend on the side
#plt.figure(1, figsize = (4,3)) # if no legend on the side
plt.figure(1, figsize = (5,3))  # if legend on the side
plt_ax = plt.subplot(111)

# Create surface of lobe width if cylinder models
if cylinder_flag:
    CS_width = plt.contourf(r_array, a_array, width_array, 10)
    cbar     = plt.colorbar()
    cbar.ax.set_ylabel(r'Lobe width / $0.5\pi$')

# XXX TEMPORARY HACK
if analytical_flag:
    #CS_width = plt.contourf(r_array, a_array, compaction_array, 10)
    #CS_width = plt.contourf(r_array, a_array, shear_array, 10)
    #CS_width = plt.contourf(r_array, a_array, shear_array - compaction_array, 10)
    CS_width = plt.contourf(r_array, a_array, shear_array / compaction_array, 10)
    cbar     = plt.colorbar()
    #cbar.ax.set_ylabel('Porosity anomaly')
    #cbar.ax.set_ylabel('Shear - compaction anomaly')
    cbar.ax.set_ylabel('Shear / compaction anomaly ratio')


# Create contours of max strain
#contours = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
contours = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
CS       = plt.contour(r_array, a_array, t_array, contours, colors = 'black', linewidth = 1.0)
manual_locations = [(3.0, 47), (7.5, 42.5), (10, 35), (12.5, 27.5), (15, 22.5), (17.5, 20)]
plt.clabel(CS, inline = 1, fontsize = 10, fmt = '%g', manual = manual_locations)

# Plot shear banding regimes as symbols
if random_flag:
    # Plot symbols for shear banding (1)
    plt.scatter(r_vals_shear, a_vals_shear, marker = 's', color = 'green', \
                edgecolor = 'black', linewidth = '1', s = 200)

    # Plot symbols for shear banding away from inclusion (2)
    plt.scatter(r_vals_some_shear, a_vals_some_shear, marker = '^', color = 'blue', \
                edgecolor = 'black', linewidth = '1', s = 200)

    # Plot symbols for no shear banding (0)
    plt.scatter(r_vals_noshear, a_vals_noshear, marker = 'o', color = 'red', \
                edgecolor = 'black', linewidth = '1', s = 200)

else:
    # Plot original data points
    plt.scatter(r_vals, a_vals, marker = 'o', color = 'black', \
                edgecolor = 'black', linewidth = '1', s = 20)

# Finish figure
# X axis
#plt.xlabel(r'$\log_{10} (r_{\zeta})$')
#plt.xlabel(r'$r_{\zeta}$')
plt.xlim(0, 22)
plt.xlabel(r'$R$')

# Y axis
plt.ylim(11, 54)
plt.ylabel(r'$\alpha$')

# Legend
legend_list = ['Bands', 'Partial bands', 'No bands']
plt_box     = plt_ax.get_position()
plt_ax.set_position([plt_box.x0, plt_box.y0, plt_box.width*0.8, plt_box.height])
plt_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5), \
              scatterpoints = 1, prop={'size':11}, labelspacing = 0.8, \
              markerscale = 1.0)

# Save figure to file
plt.savefig(fig_name, bbox_inches='tight')

# EOF plot_regimes.py
