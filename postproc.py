#!/usr/bin/env python

# ======================================================================
# Script postproc.py
#
# Does postprocessing for advection and compaction of a porous medium.
#
# Run using:
#     postproc.py <param>.cfg
#
# where <param>.cfg is a file with run parameters; an example is
# provided in the source directory. One can provide the parameter file
# in the local run directory, and rename it at will.
#
# The code uses results from main.py in simple_shear.
#
# This code only works in serial.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
# Sander Rhebergen, University of Oxford
# John Rudge, University of Cambridge
# Garth N. Wells <gnw20@cam.ac.uk>, University of Cambridge
#
# Last modified: 23 Sept 2013 by Laura Alisic
# ======================================================================

from dolfin import *
import sys, math
import core
import physics
import analysis
import datetime
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

def determine_timestep(vel):
    """Figure out time step length"""

    mesh   = vel.function_space().mesh()

    W      = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    w      = TestFunction(W)

    volume = w.cell().volume
    L      = w*sqrt(dot(vel, vel))/volume*dx
    b      = assemble(L)
    umax   = b.norm("linf")

    h_min = mesh.hmin()
    dt     = cfl*h_min/max(1.0e-6, umax)
    #dt = 0.000413

    return dt

def find_min_max(x_vals, y_vals):
    """Find minima and maxima in integrals, store as arrays through time"""

    # This needs to be done in two intervals, to make sure the two minima
    # and two maxima are found
    integral = np.array(y_vals)
    length   = integral.shape[0]

    # First minimum
    xloc1 = np.argmin(integral[:length/2])
    xmin1 = x_vals[xloc1]
    
    # Second minimum
    xloc2 = length/2 + np.argmin(integral[length/2:])
    xmin2 = x_vals[xloc2]
    
    # First maximum
    xloc1 = np.argmax(integral[:length/2])
    xmax1 = x_vals[xloc1]
    
    # Second maximum
    xloc2 = length/2 + np.argmax(integral[length/2:])
    xmax2 = x_vals[xloc2]

    return integral, xmin1, xmin2, xmax1, xmax2

# ======================================================================
# Run parameters
# ======================================================================

param_file = sys.argv[1]
param      = core.parse_param_file(param_file)

# General parameters
logname  = param['logfile']
out_freq = param['out_freq']

# FEM parameters
degree   = param['degree']

# Mesh parameters
read_mesh     = param['read_mesh']
cylinder_mesh = param['cylinder_mesh']
meshfile      = param['meshfile']
meshtype      = param['meshtype']
aspect        = param['aspect']
el            = param['el']
height        = param['height']
radius        = param['radius']

# Physics parameters
R     = param['R']
m     = param['m']
n     = param['n']
rzeta = param['rzeta']
alpha = param['alpha']

# Initial porosity parameters
initial_porosity_field = param['initial_porosity_field']
phiB      = param['phiB']
amplitude = param['amplitude']
angle_0   = math.pi * param['angle_0']
k_0       = math.pi * param['k_0']
nr_sines  = param['nr_sines']

# Time stepping parameters
tmax      = param['tmax']
cfl       = param['cfl']
dt        = param['dt']

# Input files
h5file_phi    = HDF5File("porosity.h5", "r")
h5file_vel    = HDF5File("velocity.h5", "r")
h5file_pres   = HDF5File("pressure.h5", "r")

# Output files
output_dir      = "output/"
extension       = "pvd"   # "xdmf" or "pvd"
vel_fluid_out   = File(output_dir + "melt_velocity." + extension)
vorticity_out   = File(output_dir + "pert_vorticity." + extension)
#strain_rate_out = File(output_dir + "strain_rate." + extension)

# Needed for interpolating fields without throwing an error
parameters['allow_extrapolation'] = True

# ======================================================================
# Function spaces
# ======================================================================

# Read mesh from porosity file (can do from any of the .h5 files)
mesh = Mesh()
h5file_phi.read(mesh, "mesh_file")

# Define function spaces
# Velocity
V = VectorFunctionSpace(mesh, "Lagrange", degree)

# Pressure, porosity
Q = FunctionSpace(mesh, "Lagrange", degree-1)

# Define functions
# Velocity
vel = Function(V)

# Porosity
phi = Function(Q)

# Pressure
pres = Function(Q)

# Define background velocity field due to the simple shear. This is
# later used to determine velocity perturbations in solid and fluid.
v_background = Expression(("x[1]", "0.0"))

# Background velocity field
v0 = Function(V)
v0.interpolate(v_background)

# ======================================================================
# Post-processing in time loop 
# ======================================================================

# Loop over output steps
nr_steps  = 0
strain    = []
time_step = []

while 1:
    i = nr_steps*out_freq

    try:
        # Read datasets for step i
        h5file_phi.read(phi, "porosity_%d" % i)
        h5file_vel.read(vel, "velocity_%d" % i)
        h5file_pres.read(pres, "pressure_%d" % i)
    except:
        print 'Last output step reached.'
        break

    # Determine time step length using velocity field
    dt = determine_timestep(vel)
    time_step.append(dt)         
    if nr_steps == 0: 
        strain = [0]
    else:
        strain.append(strain[nr_steps-1] + out_freq*time_step[nr_steps-1])

    print 'Step:', nr_steps, ', output step:', i, ', time step length:', time_step[nr_steps], \
          ', total strain:', strain[nr_steps]

    # Compute viscosities from porosity
    shear_visc = physics.eta(phi, param)
    bulk_visc  = physics.zeta(phi, shear_visc, param)
    perm       = physics.perm(phi, param)

    # Compute melt velocity field
    mu_f           = 1e-3 # This can be set to any desired value
    vel_fluid      = vel/phi - (perm/mu_f)*grad(pres)
    vel_fluid_proj = project(vel_fluid, V)
    vel_fluid_proj.rename("vel_fluid", "")
    vel_fluid_out << vel_fluid_proj

    # Compute perturbation vorticity field
    vort      = curl(vel - v0)
    vort_proj = project(vort, Q)
    vort_proj.rename("vort_proj", "")
    vorticity_out << vort_proj 

    # TODO: Compute strain rate field (2nd invariant)
    #strain_rate = 
    #strain_rate_proj = project(strain_rate, Q)
    #strain_rate_proj.rename("strain_rate", "")
    #strain_rate_out << strain_rate_proj


    if cylinder_mesh:
        # Computation of integrals around cylinder
        analysis.cylinder_integrals(phi, 'porosity', param, i)
        analysis.cylinder_integrals(project(div(vel)), 'compaction_rate', param, i)

    nr_steps += 1

# ======================================================================
# Preparation for figures 
# ======================================================================


# Read in integral data, reorganise into array
for j in range(nr_steps):

    i = j*out_freq
    print 'Integrals read for step', i,', strain', strain[j]

    # Integral files
    data_comp = "./output/radius_integral_compaction_rate_%s.txt" % (i)
    data_phi  = "./output/radius_integral_porosity_%s.txt" % (i)

    # Read in the integral files, store as arrays
    [x_comp, y_comp] = read_data(data_comp)
    [x_phi, y_phi]   = read_data(data_phi)

    # Find min and max of the integrals at every time step, store in vector 
    [comp_integral, comp_min1, comp_min2, comp_max1, comp_max2] = find_min_max(x_comp, y_comp)
    [phi_integral, phi_min1, phi_min2, phi_max1, phi_max2]      = find_min_max(x_phi, y_phi)

    # Store integral values in arrays
    if i == 0:
        comp_y_array    = comp_integral
        comp_min1_array = comp_min1
        comp_min2_array = comp_min2
        comp_max1_array = comp_max1
        comp_max2_array = comp_max2
        phi_y_array     = phi_integral
        phi_min1_array  = phi_min1
        phi_min2_array  = phi_min2
        phi_max1_array  = phi_max1
        phi_max2_array  = phi_max2

    else:
        comp_y_array    = np.vstack([comp_y_array, comp_integral])    
        comp_min1_array = np.hstack([comp_min1_array, comp_min1])
        comp_min2_array = np.hstack([comp_min2_array, comp_min2])
        comp_max1_array = np.hstack([comp_max1_array, comp_max1])
        comp_max2_array = np.hstack([comp_max2_array, comp_max2])
        phi_y_array     = np.vstack([phi_y_array, phi_integral])    
        phi_min1_array  = np.hstack([phi_min1_array, phi_min1])
        phi_min2_array  = np.hstack([phi_min2_array, phi_min2])
        phi_max1_array  = np.hstack([phi_max1_array, phi_max1])
        phi_max2_array  = np.hstack([phi_max2_array, phi_max2])
       
# Prepare dotted lines at original lobe positions
t_array = np.array(strain)
xmin1_nonrot = 0.25*pi*np.ones(t_array.shape[0])
xmin2_nonrot = 0.75*pi*np.ones(t_array.shape[0])
xmax1_nonrot = 1.25*pi*np.ones(t_array.shape[0])
xmax2_nonrot = 1.75*pi*np.ones(t_array.shape[0])

# ======================================================================
# Create surface figure for porosity integral 
# ======================================================================

# Create surface plot from integrals, with time
x_array = np.array(x_phi)
X, T = np.meshgrid(x_array, t_array)

# Create figure
plt.figure(1, figsize = (6,4))

# Plot surface of integral values
# TODO: Use the global min/max to determine color scale in the surface plots,
#       or manually set some sensible bounds?
phi_min = phiB - 0.05
phi_max = phiB + 0.05
delta_phi = 0.005
phi_levels = np.arange(phi_min, phi_max, delta_phi)
CF_phi = plt.contourf(X, T, phi_y_array, phi_levels, vmin = phi_min, vmax = phi_max, extend = 'both')
CF_phi.set_cmap('seismic')

# Plot dotted lines for minima and maxima at 45 degrees without rotation
plt.plot(xmin1_nonrot, t_array, '--k')
plt.plot(xmin2_nonrot, t_array, '--k')
plt.plot(xmax1_nonrot, t_array, '--k')
plt.plot(xmax2_nonrot, t_array, '--k')

# Plot minima and maxima as continuous lines, skip the first time step
#plt.plot(phi_min1_array[1:], t_array[1:], '-k')
#plt.plot(phi_min2_array[1:], t_array[1:], '-k')
#plt.plot(phi_max1_array[1:], t_array[1:], '-k')
#plt.plot(phi_max2_array[1:], t_array[1:], '-k')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel(r'Angle $\theta$')

# Y axis
plt.ylabel(r'Strain $\gamma$')

# Legend and title
plt.title('Porosity integral around the cylinder')
cbar = plt.colorbar()
cbar.set_clim(phi_min, phi_max)
c_ticks = np.arange(phi_min, phi_max, 0.01)
cbar.set_ticks(c_ticks)

# Save figure
plt.savefig('surf_phi.pdf', bbox_inches = 'tight')

# ======================================================================
# Create surface figure for compaction rate integral 
# ======================================================================

# Create surface plot from integrals, with time
x_array = np.array(x_comp)
X, T = np.meshgrid(x_array, t_array)

# Create figure
plt.figure(2, figsize = (6,4))

# Plot surface of integral values
# TODO: Use the global min/max to determine color scale in the surface plots,
#       or manually set some sensible bounds?
comp_min   = -0.5 
comp_max   = 0.5 
delta_comp = 0.05
comp_levels = np.arange(comp_min, comp_max, delta_comp)
CF_comp = plt.contourf(X, T, comp_y_array, comp_levels, vmin = comp_min, vmax = comp_max, extend = 'both')
CF_comp.set_cmap('seismic')

# Plot dotted lines for minima and maxima at 45 degrees without rotation
plt.plot(xmin1_nonrot, t_array, '--k')
plt.plot(xmin2_nonrot, t_array, '--k')
plt.plot(xmax1_nonrot, t_array, '--k')
plt.plot(xmax2_nonrot, t_array, '--k')

# Plot minima and maxima as continuous lines, skip the first time step
#plt.plot(comp_min1_array[1:], t_array[1:], '-k')
#plt.plot(comp_min2_array[1:], t_array[1:], '-k')
#plt.plot(comp_max1_array[1:], t_array[1:], '-k')
#plt.plot(comp_max2_array[1:], t_array[1:], '-k')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel(r'Angle $\theta$')

# Y axis
plt.ylabel(r'Strain $\gamma$')

# Legend and title
plt.title('Compaction rate integral around the cylinder')
cbar = plt.colorbar()
cbar.set_clim(comp_min, comp_max)
c_ticks = np.arange(comp_min, comp_max, 0.2)
cbar.set_ticks(c_ticks)
#cbar.set_label(r'$\phi$')

# Save figure
plt.savefig('surf_comp.pdf', bbox_inches = 'tight')


# ======================================================================
# Prepare for line plots
# ======================================================================

# Pick several time steps to plot, find corresponding output step
plot_times = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0])
plot_steps = []
for j, step in enumerate(plot_times):
    # Only do if the requested time is smaller than the total model runtime;
    # margin parameter ensures that a final time very close to but smaller 
    # than a plot time is still taken into account
    margin = 0.001
    if step <= (t_array.max() + 0.001):
        # Find index of strain step closest to plot time requested
        index = np.argmin(abs(t_array-step))
        plot_steps.append(index)
    else:
        break

# Clean exit if plot times are all larger than max model time
if len(plot_steps) < 2:
   print 'No results to be plotted with selected output times.'
   sys.exit()

# List of colors to use
step_color = ['Black', 'Blue', 'Green', 'Yellow', 'DarkOrange', 'Red', \
              'Magenta', 'Cyan', 'Lime', 'Orange', 'FireBrick', 'Indigo']

# ======================================================================
# Create line plots for porosity integrals
# ======================================================================

# Create figure for porosity
plt.figure(3, figsize = (7,4))
ax = plt.subplot(111)

# Loop over steps to be plotted
legend_list = []
for k, step in enumerate(plot_steps):
    print 'Plotting porosity integral at strain ', strain[int(step)]
    plt.plot(x_array, phi_y_array[int(step),:], step_color[k])
    legend_text = r'$\gamma$ = ' + str(plot_times[k])
    legend_list.append(legend_text)
    max_plot_step = int(step)

# Plot dotted black line at compaction rate = 0.0
background_array = phiB*np.ones(x_array.shape[0])
plt.plot(x_array, background_array, '--k')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel('Angle')

# Y axis
plt.ylim(phi_y_array[:max_plot_step].min()*0.9, phi_y_array[:max_plot_step].max()*1.1)
plt.ylabel('Porosity')

# Legend and title
plt.title('Porosity integrals around the cylinder')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig('porosity_integrals.pdf', bbox_inches='tight')

# ======================================================================
# Create line plots for compaction rate integrals
# ======================================================================

# Create figure for compaction rate
plt.figure(4, figsize = (7,4))
ax = plt.subplot(111)

# Loop over steps to be plotted
for k, step in enumerate(plot_steps):
    print 'Plotting compaction rate integral at strain ', strain[int(step)]
    plt.plot(x_array, comp_y_array[int(step),:], step_color[k])

# Plot dotted black line at compaction rate = 0.0
zero_array = np.zeros(x_array.shape[0])
plt.plot(x_array, zero_array, '--k')

# X axis
plt.xlim(0, 2*np.pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],\
           [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
plt.xlabel('Angle')

# Y axis
plt.ylim(comp_y_array[:max_plot_step][:].min()*1.1, comp_y_array[:max_plot_step][:].max()*1.1)
plt.ylabel('Compaction rate')

# Legend and title
plt.title('Compaction rate integrals around the cylinder')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig('compaction_integrals.pdf', bbox_inches='tight')

# EOF postproc.py
