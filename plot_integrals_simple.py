#!/usr/bin/env python

# ======================================================================
# Script plot_integrals_simple.py
#
# Plots integrals in one plot; defined simply by listing file names.
# Currently only integrals at t = 0 are plotted.
#
# Run using:
#     python plot_integrals_simple.py
#
# The code uses results from postproc_3D.py or postproc.py.
#
# This code only works in serial.
#
# Author:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
#
# Last modified: 16 Apr 2014 by Laura Alisic
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
title = 'N10'

# List of integrals to plot
model_list = ['../num_N10', \
              '../analytic_N10']

# List of colors to use
color_list = ['Black', 'Blue', 'Green', 'Red', 'Cyan', 'Magenta']

# Text strings for legend
legend_list = ['Numerical', 'Analytical']

# Figure output names
pf_fig_name   = 'pressure_integrals_%s.pdf' % (title)
pc_fig_name   = 'compaction_pressure_integrals_%s.pdf' % (title)
comp_fig_name = 'compaction_rate_integrals_%s.pdf' % (title)

# Figure IDs
pf_id   = 1
pc_id   = 2
comp_id = 3

# ======================================================================
# Prepare integral figures
# ======================================================================

# Create figure for pressure
plt.figure(pf_id, figsize = (5,3))
pf_ax  = plt.subplot(111)
pf_min = 0.0
pf_max = 0.0

# Create figure for compaction pressure
plt.figure(pc_id, figsize = (5,3))
pc_ax  = plt.subplot(111)
pc_min = 0.0
pc_max = 0.0

# Create figure for compaction rate
plt.figure(comp_id, figsize = (5,3))
comp_ax  = plt.subplot(111)
comp_min = 0.0
comp_max = 0.0
 
for j, model in enumerate(model_list):

    print 'Plotting...'

    # Figure out file names to read
    pf_name   = '%s/output/radius_integral_pressure_0.txt' % (model)
    pc_name   = '%s/output/radius_integral_compaction_pressure_0.txt' % (model)
    comp_name = '%s/output/radius_integral_compaction_rate_0.txt' % (model)

    [x_pf, y_pf]     = read_data(pf_name) 
    [x_pc, y_pc]     = read_data(pc_name) 
    [x_comp, y_comp] = read_data(comp_name) 
                
    # Plot pressure integrals 
    plt.figure(pf_id)
    plt.plot(x_pf, y_pf, color_list[j])

    # Plot compaction pressure integrals 
    plt.figure(pc_id)
    plt.plot(x_pc, y_pc, color_list[j])

    # Plot compaction rate integrals 
    plt.figure(comp_id)
    plt.plot(x_comp, y_comp, color_list[j])

    # Save min/max y values to determine y-axis range
    pf_min = min(pf_min, min(y_pf))
    pf_max = max(pf_max, max(y_pf))
    pc_min = min(pc_min, min(y_pc))
    pc_max = max(pc_max, max(y_pc))
    comp_min = min(comp_min, min(y_comp))
    comp_max = max(comp_max, max(y_comp))

# ======================================================================
# Finish pressure plot 
# ======================================================================

# Plot dotted black line at zero
plt.figure(pf_id)
pf_background_array = np.zeros(len(x_pf))
plt.plot(x_pf, pf_background_array, '--k')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel('Angle')

# Y axis
plt.ylim(pf_min*0.90, pf_max*1.03)
plt.ylabel('Pressure')

# Legend and title
title_string = '%s: inclusion radius 0.2' % (title)
plt.title(title_string)
pf_box = pf_ax.get_position()
pf_ax.set_position([pf_box.x0, pf_box.y0, pf_box.width*0.8, pf_box.height])
pf_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(pf_fig_name, bbox_inches='tight')

# ======================================================================
# Finish compaction pressure plot 
# ======================================================================

# Plot dotted black line at zero
plt.figure(pc_id)
pc_background_array = np.zeros(len(x_pc))
plt.plot(x_pc, pc_background_array, '--k')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel('Angle')

# Y axis
plt.ylim(pc_min*0.90, pc_max*1.03)
plt.ylabel('Pressure')

# Legend and title
plt.title(title_string)
pc_box = pc_ax.get_position()
pc_ax.set_position([pc_box.x0, pc_box.y0, pc_box.width*0.8, pc_box.height])
pc_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(pc_fig_name, bbox_inches='tight')

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
plt.ylim(comp_min*1.2, comp_max*1.05)
plt.ylabel('Compaction rate')

# Legend and title
plt.title(title_string)
comp_box = comp_ax.get_position()
comp_ax.set_position([comp_box.x0, comp_box.y0, comp_box.width*0.8, comp_box.height])
comp_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(comp_fig_name, bbox_inches='tight')

# Finish off
plt.close(pf_id)
plt.close(pc_id)
plt.close(comp_id)

print 'Done!'

# EOF plot_integrals_simple.py
