#!/usr/bin/env python3

# ======================================================================
# Script plot_integrals_3D.py
#
# Plots integrals in one plot; defined simply by listing file names.
# Currently only integrals at t = 0 are plotted.
#
# Run using:
#     python3 plot_integrals_3D.py
#
# The code uses results from postproc_3D.py or postproc.py.
#
# This code only works in serial.
#
# Author:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
#
# Last modified: 1 April 2015 by Laura Alisic
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

phi0  = 0.05

# output for same model at times 0, 0.1, 0.2, 0.3, 0.5 (, 1.0 if available)
#step_list  = [0, 6, 12, 18, 30] # output_freq = 10
#step_list  = [0, 6, 12, 18] # output_freq = 10
#step_list  = [0, 3, 6, 9] 
step_list  = [12, 24, 24, 23] 

#model = ['.', '.', '.', '.', '.']

model = ['../../uniform_inclusion/uniform_alpha0_R5', \
         '../uniform_inc_alpha0_n2_R5', \
         '../uniform_inc_alpha0_n4_R5', \
         '../uniform_inc_alpha0_n6_R5']

# List of colors to use
color_list = ['Black', 'Blue', 'Green', 'Red', 'Cyan', 'Magenta']

# Text strings for legend
#legend_list = [r'$R$ = 1.7', r'$R$ = 5.0', r'$R$ = 10.0', r'$R$ = 20.0']
#legend_list = [r'$\alpha$ = 0', r'$\alpha$ = 15', r'$\alpha$ = 28']#, \
#               r'$\alpha$ = 50']
legend_list = [r'$n$ = 1', r'$n$ = 2', r'$n$ = 4', r'$n$ = 6']
#legend_list = [r'$t$ = 0', r'$t$ = 0.1', r'$t$ = 0.2', r'$t$ = 0.3', r'$t$ = 0.5']
#legend_list = [r'$t$ = 0', r'$t$ = 0.25', r'$t$ = 0.5', r'$t$ = 1.0'] #, r'$t$ = 2.0']

# Figure output names
phi_fig_name  = 'porosity_integrals_alpha0_R5_strain0_4.pdf'
comp_fig_name = 'compaction_rate_integrals_alpha0_R5_strain0_4.pdf'

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

    print('Plotting step ', step)

    # Figure out file names to read
    phi_name   = '%s/output/radius_integral_porosity_%d.txt' % (model[j], step)
    comp_name = '%s/output/radius_integral_compaction_rate_%d.txt' % (model[j], step)

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

# Finish porosity plot 
# Plot dotted black line at background porosity
plt.figure(phi_id)
phi_background_array = phi0*np.ones(len(x_phi))
plt.plot(x_phi, phi_background_array, '--k', label = '_nolegend_')


# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel('Angle')

# Y axis
plt.ylim(phi_min*0.90, phi_max*1.03)
plt.ylabel('Porosity')

# Legend and title
#title_string = r'$\alpha$ = %g, $r_{\zeta}$ = %g' % (alpha[0], rzeta[0])
#title_string = r'$\alpha$ = %g, $\gamma$ = 0.1' % (alpha[0])
#title_string = r'$r_{\zeta}$ = %g, $\gamma$ = 0.1' % (rzeta[0])
#plt.title(title_string)
phi_box = phi_ax.get_position()
phi_ax.set_position([phi_box.x0, phi_box.y0, phi_box.width*0.8, phi_box.height])
phi_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(phi_fig_name, bbox_inches='tight')

# ======================================================================
# Finish compaction rate plot 
# ======================================================================

# Finish compaction rate plot 
# Plot dotted black line at compaction rate = 0.0
plt.figure(comp_id)
comp_zero_array = np.zeros(len(x_comp))
plt.plot(x_comp, comp_zero_array, '--k', label = '_nolegend_')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel('Angle')

# Y axis
plt.ylim(comp_min*1.2, comp_max*1.05)
plt.ylabel('Compaction rate')

# Legend and title
#plt.title(title_string)
comp_box = comp_ax.get_position()
comp_ax.set_position([comp_box.x0, comp_box.y0, comp_box.width*0.8, comp_box.height])
comp_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(comp_fig_name, bbox_inches='tight')


# Finish off
plt.close(phi_id)
plt.close(comp_id)

print('Done!')

# EOF plot_integrals_3D.py
