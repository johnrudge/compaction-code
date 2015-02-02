#!/usr/bin/env python

# ======================================================================
# Script plot_integrals_3D.py
#
# Plots integrals in one plot; defined simply by listing file names.
# Currently only integrals at t = 0 are plotted.
#
# Run using:
#     python plot_integrals_3D.py
#
# The code uses results from postproc_3D.py or postproc.py.
#
# This code only works in serial.
#
# Author:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
#
# Last modified: 2 Feb 2015 by Laura Alisic
# ======================================================================

import sys, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.ioff()

# ======================================================================
# Functions
# ======================================================================

def read_data(data_file):
    """Read in data from integral files, store in numpy arrays"""

    file_in = open(data_file,"r")

    # Arrays to store data in for further plotting etc
    x_array = []
    y_array = []  
 
    line_nr = 1
    for line in file_in.readlines():
        xval, yval = line.split()
         
        x_array.append(float(xval))
        y_array.append(float(yval))

    file.close(file_in)

    # Append value of 0pi at 2pi
    x_array.append(2.0*np.pi)
    y_array.append(y_array[0])

    return x_array, y_array

# ======================================================================
# Parameters
# ======================================================================

# Title string
title = 'N20_test'
title_string = '%s' % (title)

# List of integrals to plot
#model_list = ['../analytic_N30_r0p05', \
#              '../num_N30_r0p05']
model     = '.'
step_list = [0, 1, 2]

# List of colors to use
color_list = ['Black', 'Blue', 'Green', 'Red', 'Cyan', 'Magenta']

# Text strings for legend
legend_list = ['outstep 0', 'outstep 1', 'outstep 2']

# Figure output names
phi_fig_name  = 'porosity_integrals_%s.pdf' % (title)
comp_fig_name = 'compaction_rate_integrals_%s.pdf' % (title)

# Figure IDs
phi_id  = 1
comp_id = 2

# ======================================================================
# Prepare integral figures
# ======================================================================

# Create figure for porosity
plt.figure(phi_id, figsize = (5,3))
phi_ax  = plt.subplot(111)
phi_min = 0.05
phi_max = 0.05

# Create figure for compaction rate
plt.figure(comp_id, figsize = (5,3))
comp_ax  = plt.subplot(111)
comp_min = 0.0
comp_max = 0.0
 
#for j, model in enumerate(model_list):
for j, step in enumerate(step_list):

    print 'Plotting step ', step

    # Figure out file names to read
    phi_name   = '%s/output/radius_integral_porosity_%d.txt' % (model, step)
    comp_name = '%s/output/radius_integral_compaction_rate_%d.txt' % (model, step)

    [x_phi, y_phi]   = read_data(phi_name) 
    [x_comp, y_comp] = read_data(comp_name) 
                
    # Plot pressure integrals 
    plt.figure(phi_id)
    plt.plot(x_phi, y_phi, color_list[j])

    # Plot compaction rate integrals 
    plt.figure(comp_id)
    plt.plot(x_comp, y_comp, color_list[j])

    # Save min/max y values to determine y-axis range
    phi_min = min(phi_min, min(y_phi))
    phi_max = max(phi_max, max(y_phi))
    comp_min = min(comp_min, min(y_comp))
    comp_max = max(comp_max, max(y_comp))

# ======================================================================
# Finish porosity plot 
# ======================================================================

# Plot dotted black line at background porosity
plt.figure(phi_id)
phi_background_array = np.ones(len(x_phi)) * 0.05
plt.plot(x_phi, phi_background_array, '--k')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel('Angle')

# Y axis
plt.ylim(phi_min*0.80, phi_max*1.20)
plt.ylabel('Porosity')

# Legend and title
plt.title(title_string)
phi_box = phi_ax.get_position()
phi_ax.set_position([phi_box.x0, phi_box.y0, phi_box.width*0.8, phi_box.height])
phi_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(phi_fig_name, bbox_inches='tight')

# ======================================================================
# Finish compaction rate plot 
# ======================================================================

# Plot dotted black line at compaction rate = 0.0
plt.figure(comp_id)
comp_zero_array = np.zeros(len(x_comp))
plt.plot(x_comp, comp_zero_array, '--k')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel('Angle')

# Y axis
plt.ylim(comp_min*1.2, comp_max*1.20)
plt.ylabel('Compaction rate')

# Legend and title
plt.title(title_string)
comp_box = comp_ax.get_position()
comp_ax.set_position([comp_box.x0, comp_box.y0, comp_box.width*0.8, comp_box.height])
comp_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(comp_fig_name, bbox_inches='tight')

# Finish off
plt.close(phi_id)
plt.close(comp_id)

print 'Done!'

# EOF plot_integrals_3D.py
