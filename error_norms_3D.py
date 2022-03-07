#!/usr/bin/env python3

# ======================================================================
# error_norms_3D.py
#
# Computes error norms for numerical models with respect to analytical
# solutions of a 3-D torsion experiment with a spherical inclusion.
#
# Results are plotted vs inclusion radius.
#
# Requires *.h5 output for the numerical and analytical solutions;
# fields are tailored to John's and Sander's codes.
#
# Run by using:
#     python3 error_norms_3D.py
#
# Author:
# Laura Alisic, University of Cambridge
#
# Last modified: 28 May 2015 by Laura Alisic
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

def compute_u_torsion(V):
    '''Compute background velocity according to torsion'''

    class USolution(Expression):

        def eval(self, value, x):

            r = math.sqrt(x[0]*x[0] + x[1]*x[1])
            beta = math.atan2(x[1],x[0])

            value[0] = -r*(x[2]-0.5)*math.sin(beta)
            value[1] = r*(x[2]-0.5)*math.cos(beta)
            value[2] = 0.0

        def value_shape(self):
            return (3,)

    u_background = USolution()
    u0 = Function(V)
    u0.interpolate(u_background)

    return u0

def compute_errors(radius, pf_an, pf_num, pc_an, pf_nu, u0, u_an, u_num):
    '''Compute error norms between numerical and analytical solutions'''

    # Compute error norms
    info("**** Computing error norms...")

    # Domain volume
    height = 1.0
    cylinder_radius = 1.0
    volume = math.pi*height*cylinder_radius**2 - math.pi*radius**2

    # Pressure error
    mean_pf     = assemble(pf_num*dx) / volume
    info("---- Mean numerical pressure is %g" % (mean_pf))
    shifted_pf  = project(pf_num - mean_pf, Q)
    an_norm_pf  = norm(pf_an)
    num_norm_pf = norm(shifted_pf)
    error_pf    = (errornorm(pf_an, shifted_pf)) / an_norm_pf
    info("---- Pressure L2 norms: analytical %g, numerical %g, error %g" \
         % (an_norm_pf, num_norm_pf, error_pf))

    # Compaction pressure error
    mean_pc     = assemble(pc_num*dx) / volume
    info("---- Mean numerical compaction pressure is %g" % (mean_pc))
    shifted_pc  = project(pc_num - mean_pc, Q)
    an_norm_pc  = norm(pc_an)
    num_norm_pc = norm(shifted_pc)
    error_pc    = (errornorm(pc_an, shifted_pc)) / an_norm_pc
    info("---- Compaction pressure L2 norms: analytical %g, numerical %g, error %g" \
         % (an_norm_pc, num_norm_pc, error_pc))

    # Velocity error
    an_norm_u  = norm(u_an)
    num_norm_u = norm(u_num)
    error_u    = (errornorm(u_an, u_num)) / an_norm_u
    info("---- Velocity L2 norms: analytical %g, numerical %g, error %g" \
         % (an_norm_u, num_norm_u, error_u))

    # Velocity perturbation error: substract background torsioni
    du_an = u_an - u0
    du_an_proj = project(du_an)
    an_norm_du  = norm(du_an_proj)
    
    du_num = u_num - u0
    du_num_proj = project(du_num)
    num_norm_du = norm(du_num_proj)

    error_du    = (errornorm(du_an_proj, du_num_proj)) / an_norm_du
    info("---- Velocity perturbation L2 norms: analytical %g, numerical %g, error %g" \
         % (an_norm_du, num_norm_du, error_du))

    # Compaction rate error
    comp_an       = project(div(u_an))
    comp_num      = project(div(u_num))
    an_norm_comp  = norm(comp_an)
    num_norm_comp = norm(comp_num)
    error_comp    = (errornorm(comp_an, comp_num)) / an_norm_comp
    info("---- Compaction rate L2 norms: analytical %g, numerical %g, error %g" \
         % (an_norm_comp, num_norm_comp, error_comp))

    return error_pf, error_pc, error_u, error_du, error_comp


# ======================================================================
# Parameters
# ======================================================================

# Make sure that these lists are the same length: the first analytical
# model is compared with the first numerical, and so on. The pairs have
# to be defined on the same mesh.
an_model_list  = ['new_analytic_N50_r02', 'new_analytic_N50_r01', 'new_analytic_N50_r005']
num_model_list = ['new_num_N50_r02', 'new_num_N50_r01', 'new_num_N50_r005']

# Figure names
lin_fig_name = 'error_norms_linear_N50_vel_perturb.pdf'
log_fig_name = 'error_norms_loglog_N50_vel_perturb.pdf'

# List with inclusion radii corresponding to the above model pairs;  also 
# needs to be the same length as the model list vectors.
radius = [0.2, 0.1, 0.05]

# Degree of vector function space
degree = 2

# Needed for interpolating fields without throwing an error
parameters['allow_extrapolation'] = True

# MPI command needed for HDF5
comm = MPI.comm_world

# ======================================================================
# Loop over models
# ======================================================================

error_pf_vals   = []  # Pressure error
error_pc_vals   = []  # Compaction pressure error
error_u_vals    = []  # Velocity error
error_du_vals   = [] # Velocity perturbation error
error_comp_vals = []  # Compaction rate error

for i, model in enumerate(an_model_list):
    print('Models:', an_model_list[i], 'and', num_model_list[i])

    # Define files
    info("**** Defining input files...")

    # Input files from analytical code by John
    h5file_u_an   = HDF5File(comm, ("../%s/output/velocity.h5" % (an_model_list[i])), "r")
    h5file_pf_an  = HDF5File(comm, ("../%s/output/pressure.h5" % (an_model_list[i])), "r")
    h5file_pc_an  = HDF5File(comm, ("../%s/output/compaction_pressure.h5" % (an_model_list[i])), "r")

    # Input files from numerical code by Sander
    h5file_u_num  = HDF5File(comm, ("../%s/velocity_0.h5" % (num_model_list[i])), "r")
    h5file_pf_num = HDF5File(comm, ("../%s/pressure_pf_0.h5" % (num_model_list[i])), "r")
    h5file_pc_num = HDF5File(comm, ("../%s/pressure_pc_0.h5" % (num_model_list[i])), "r")

    # Read mesh from file
    #mesh = Mesh()
    #h5file_pf_num.read(mesh, "mesh_file", False)

    # For each pair of num/an models, define function spaces
    info("**** Defining function spaces...")

    # Velocity
    #V = VectorFunctionSpace(mesh, "Lagrange", degree)

    # Pressure, porosity
    #Q = FunctionSpace(mesh, "Lagrange", degree-1)

    # Define functions
    # Velocity
    #u_an  = Function(V)
    #u_num = Function(V)

    # Pressures
    #pf_an  = Function(Q)
    #pc_an  = Function(Q)
    #pf_num = Function(Q)
    #pc_num = Function(Q)

    # Read files
    info("**** Reading files...")

    # Files for analytical solution
    #h5file_u_an.read(u_an, "velocity")
    #h5file_pf_an.read(pf_an, "pressure")
    #h5file_pc_an.read(pc_an, "compaction_pressure")

    # Files for numerical solution
    #h5file_u_num.read(u_num, "velocity")
    #h5file_pf_num.read(pf_num, "pressure_pf")
    #h5file_pc_num.read(pc_num, "pressure_pc")

    # Background velocity, according to the torsion Dirichlet BC on the cylinder
    #u0 = compute_u_torsion(V)
 
    # Compute errors
    #[error_pf, error_pc, error_u, error_du, error_comp] = compute_errors(radius[i], pf_an, pf_num, pc_an, pc_num, u0, u_an, u_num)

    # Store error data in array
    #error_pf_vals.append(error_pf)
    #error_pc_vals.append(error_pc)
    #error_u_vals.append(error_u)
    #error_du_vals.append(error_du)
    #error_comp_vals.append(error_comp)


# For direct plotting: Models N20, N20_r0p1, N20_rp05
#error_pf_vals   = [ 0.858583, 0.51708, 0.415783 ]
#error_pc_vals   = [ 0.188552, 0.0430007, 0.050721 ]
#error_u_vals    = [ 0.0180874, 0.00230316, 0.000286238 ]
#error_comp_vals = [ 0.194275, 0.0441826, 0.0280429 ]

# For direct plotting: Models N30, N30_r0p1, N30_rp05
error_pf_vals   = [ 0.858752, 0.51695, 0.41648 ] 
error_pc_vals   = [ 0.189391, 0.0423787, 0.0295484 ]
error_u_vals    = [ 0.0180833, 0.0023012, 0.000285142 ]
error_du_vals   = [ 0.517893, 0.354076, 0.240777]
error_comp_vals = [ 0.1929, 0.0434048, 0.0181757 ]

# For direct plotting: Models N50_r02, N50_r01, N50_r005
#error_pf_vals   = [ ] 
#error_pc_vals   = [ ]
#error_u_vals    = [ ]
#error_du_vals   = [ ]
#error_comp_vals = [ ]

# ======================================================================
# Plot results in linear space
# ======================================================================

# Start figure
lin_id       = 1
plt.figure(lin_id, figsize = (5,3))
lin_ax       = plt.subplot(111)

# Plot data points
plt.plot(radius, error_pf_vals, marker = 'o', color = 'black')
plt.plot(radius, error_pc_vals, marker = 'x', color = 'blue')
plt.plot(radius, error_du_vals, marker = '+', color = 'red')
#plt.plot(radius, error_comp_vals, marker = 's', color = 'red')

# X axis
plt.xlim([0, 0.22])
plt.xlabel(r'Inclusion radius')

# Y axis
lin_min = min(min([error_pf_vals, error_pc_vals, error_du_vals, error_comp_vals]))
lin_max = max(max([error_pf_vals, error_pc_vals, error_du_vals, error_comp_vals]))
plt.ylim(lin_min*0.90, lin_max*1.10)
plt.ylabel(r'$L_2$ error')

# Legend
legend_list = [r'$p_f$', r'$p_c$', r'$\Delta \mathbf{u}_s$']
lin_box     = lin_ax.get_position()
lin_ax.set_position([lin_box.x0, lin_box.y0, lin_box.width*0.8, lin_box.height])
lin_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(lin_fig_name, bbox_inches='tight')


# ======================================================================
# Plot results in loglog space
# ======================================================================

# Start figure
log_id       = 2
plt.figure(log_id, figsize = (5,3))
log_ax       = plt.subplot(111)

# Plot data points
plt.loglog(radius, error_pf_vals, marker = 'o', color = 'black')
plt.loglog(radius, error_pc_vals, marker = 'x', color = 'blue')
plt.loglog(radius, error_du_vals, marker = '+', color = 'red')
#plt.loglog(radius, error_comp_vals, marker = 's', color = 'red')

# X axis
plt.xlim([3e-2, 3e-1])
plt.xticks([0.05, 0.10, 0.20], \
           ['0.05', '0.10', '0.20'])
plt.xlabel(r'Inclusion radius')

# Y axis
#log_min = min(min([error_pf_vals, error_pc_vals, error_du_vals, error_comp_vals]))
#log_max = max(max([error_pf_vals, error_pc_vals, error_du_vals, error_comp_vals]))
#plt.ylim(log_min*0.5, log_max*2.0)
plt.ylim([1e-2, 1.5])
plt.ylabel(r'$L_2$ error')

# Legend
legend_list = [r'$p_f$', r'$p_c$', r'$\Delta \mathbf{u}_s$']
log_box     = log_ax.get_position()
log_ax.set_position([log_box.x0, log_box.y0, log_box.width*0.8, log_box.height])
log_ax.legend(legend_list, loc = 'center left', bbox_to_anchor = (1, 0.5))

# Save figure to file
plt.savefig(log_fig_name, bbox_inches='tight')


info("Done!")

# EOF error_norms_3D.py
