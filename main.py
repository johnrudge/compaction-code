#!/usr/bin/env python3

# ======================================================================
# Script main.py
#
# Computes advection and compaction of a porous medium.
#
# Run using:
#     main.py <param>.cfg
#
# where <param>.cfg is a file with run parameters; an example is
# provided in the source directory. One can provide the parameter file
# in the local run directory, and rename it at will.
#
# When a mesh is generated for the cylinder option, gmsh is required (only
# works in serial!!).
# Constructive solid geometry is another option, but at the moment the
# resolution cannot be adjusted.
#
# The following output is generated:
#   pvd or xmdf files for quick visualization
#   h5 files for further postprocessing
#
# Further analysis and integrals are done using postproc.py 
#
# The code will exit when the minimum dimensional porosity becomes non-
# physical at < 0.0 or the maximum at > 1.0.
#
# Authors:
# Laura Alisic <la339@cam.ac.uk>, University of Cambridge
# Sander Rhebergen, University of Oxford
# John Rudge, University of Cambridge
# Garth N. Wells <gnw20@cam.ac.uk>, University of Cambridge
#
# ======================================================================

from mpi4py import MPI
from dolfinx.fem import Function, Constant, dirichletbc, FunctionSpace, Expression, locate_dofs_geometrical, locate_dofs_topological, form
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.mesh import create_rectangle, locate_entities_boundary, meshtags, DiagonalType
from dolfinx.la import Norm
from dolfinx.io import VTKFile, XDMFFile
from dolfinx_mpc import MultiPointConstraint, LinearProblem
from test_nonlinear_assembly import NonlinearMPCProblem, NewtonSolverMPC
from ufl import sqrt, inner, sym, dot, div, dx, grad, TrialFunction, TestFunction, TrialFunctions, TestFunctions, CellDiameter, lhs, rhs, split, VectorElement, FiniteElement, MixedElement, Measure
import numpy as np
import sys, math
import core
import physics
import analysis
import mesh_gen
import mesh_gen_uniform
import datetime

def porosity_forms(V, phi0, u, dt):
    """Return forms for porosity transport"""
    phi1    = TrialFunction(V)
    w       = TestFunction(V)
    phi_mid = 0.5*(phi1 + phi0)
    
    F = w*(phi1 - phi0 + dt*(dot(u, grad(phi_mid)) - (1.0 - phi_mid)*div(u)))*dx

    # SUPG stabilisation term
    h_SUPG   = CellDiameter(mesh)
    residual = phi1 - phi0 + dt * (dot(u, grad(phi_mid)) - div(grad(phi_mid)))
    unorm    = sqrt(dot(u, u))
    aval     = 0.5*h_SUPG*unorm
    keff     = 0.5*((aval - 1.0) + abs(aval - 1.0))
    stab     = (keff / (unorm * unorm)) * dot(u, grad(w)) * residual * dx
    F       += stab

    return lhs(F), rhs(F)


def stokes_forms(W, phi, dt, param, cylinder_mesh):
    """Return forms for Stokes-like problem"""

    if cylinder_mesh:
        (v, q, lam) = TestFunctions(W)
        u, p, omega = split(U)
    else:
        (v, q) = TestFunctions(W)
        u, p   = split(U)

    srate      = physics.strain_rate(u)
    shear_visc = physics.eta(phi, srate, param)
    bulk_visc  = physics.zeta(phi, shear_visc, param)
    perm       = physics.perm(phi, param)
    F          = 2.0*shear_visc*inner(sym(grad(u)), sym(grad(v)))*dx \
                 + (rzeta*bulk_visc - shear_visc*2.0/3.0)*div(u)*div(v)*dx \
                 - p*div(v)*dx - q*div(u)*dx \
                 - (R*R/(rzeta + 4.0/3.0))*perm*dot(grad(p), grad(q))*dx

    # Stokes source term -- will be zero for now
    f  = Constant(W.mesh, (0.0, 0.0))
    F -= dot(f, v)*dx

    # Adjustment of Stokes-type equation for cylinder using Lagrange
    # multiplier term
    if cylinder_mesh:
        # Nitsche's method rather than ordinary penalty term

        # Nitsche's method coefficient (what should this be!?)
        nitsche_fac = 10.0

        tu = physics.traction(phi, p, u, param, W.mesh())
        tv = physics.traction(phi, q, v, param, W.mesh())

        # vector for velocity around cylinder
        vcyl  = Expression(("-x[1]", "x[0]"), degree = 1)

        # This is what we want to be zero on the cylinder
        u_dirichlet = u - omega*vcyl

        h_nitsche   = CellDiameter(mesh)

        # Nitsche's method for v=omega cross x on cylinder
        F += ((nitsche_fac/h_nitsche)*dot(u_dirichlet, v) \
                  - dot(u_dirichlet, tv) - dot(v, tu))*ds_cylinder

        # Lagrange multiplier term for zero torque on cylinder
        F += lam*dot(vcyl, tu)*ds_cylinder

    return F

# ======================================================================
# Run parameters
# ======================================================================

param_file = sys.argv[1]
param   = core.parse_param_file(param_file)

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
stress_exp = param['stress_exp']

# Initial porosity parameters
initial_porosity_field = param['initial_porosity_field']
read_initial_porosity  = param['read_initial_porosity']
initial_porosity_in  = param['initial_porosity_in']
phiB      = param['phiB']
amplitude = param['amplitude']
angle_0   = math.pi * param['angle_0']
k_0       = math.pi * param['k_0']
nr_sines  = param['nr_sines']

# Time stepping parameters
tmax      = param['tmax']
cfl       = param['cfl']
dt        = param['dt']

# MPI command needed for HDF5
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Output files for quick visualisation
output_dir     = "output/"
extension      = "pvd"   # "pvd"
initial_porosity_out = VTKFile(comm, output_dir + "initial_porosity." + extension, "w")
velocity_out   = VTKFile(comm, output_dir + "velocity." + extension, "w")
vel_pert_out   = VTKFile(comm, output_dir + "velocity_perturbations." + extension, "w")
pressure_out   = VTKFile(comm, output_dir + "pressure." + extension, "w")
porosity_out   = VTKFile(comm, output_dir + "porosity." + extension, "w")
shear_visc_out = VTKFile(comm, output_dir + "shear_viscosity." + extension, "w")
bulk_visc_out  = VTKFile(comm, output_dir + "bulk_viscosity." + extension, "w")
perm_out       = VTKFile(comm, output_dir + "permeability." + extension, "w")
compaction_out = VTKFile(comm, output_dir + "compaction." + extension, "w")
num_ds_dt_out  = VTKFile(comm, output_dir + "ds_dt." + extension, "w")
divU_out       = VTKFile(comm, output_dir + "div_u." + extension, "w")
strain_rate_out = VTKFile(comm, output_dir + "strain_rate." + extension, "w")

## Output files for further postprocessing
#h5file_phi     = HDF5File(comm, "porosity.h5", "w")
#h5file_vel     = HDF5File(comm, "velocity.h5", "w")
#h5file_pres    = HDF5File(comm, "pressure.h5", "w")

# Initialise logfile
logfile = open(logname, "w", encoding="utf-8")
rank = comm.Get_rank()
if rank == 0:
    logfile.write(str(datetime.datetime.now()))

# Print params to logfile
if rank == 0:
    logfile.write("\n\nRun parameters:\n")
    for item in sorted(list(param.keys()), key=str.lower):
        logfile.write("%s = %s\n" % (item, param[item]))

# ======================================================================
# Mesh
# ======================================================================

# Create mesh
if meshtype == "left/right":
    diagonal = DiagonalType.left_right
elif meshtype == "left":
    diagonal = DiagonalType.left
else:
    diagonal = DiagonalType.right

if read_mesh:
    print("**** Reading mesh file: %s" % meshfile)
    #mesh = Mesh(meshfile)
    with XDMFFile(comm, meshfile, "r") as xdmf:
       mesh = xdmf.read_mesh(name="Grid")
    
else:
    print("**** Generating mesh . . . ")
    if cylinder_mesh:

        # Create a mesh with gmsh
        # NOTE: running this in parallel throws an error about lifeline lost.
        comm.Barrier()
        #mesh_gen_uniform.cylinder_mesh_gen(filename=meshfile, \
        mesh_gen.cylinder_mesh_gen(filename=meshfile, \
                                   aspect=aspect, \
                                   N=el, \
                                   h=height, \
                                   rel_radius=(radius/height))
        comm.Barrier()
        mesh = Mesh(meshfile)
    else:
        mesh = create_rectangle(comm, \
                        [np.array([0, 0]), np.array([aspect*height, height])], \
                        [int(aspect*el), int(el)], diagonal=diagonal)

# Smallest element size. Used to determine time step
# mesh properties
tdim = mesh.topology.dim
fdim = tdim-1
num_cells = mesh.topology.index_map(tdim).size_local    # number of cells in the mesh
h = mesh.h(tdim, range(num_cells)) # get the cell size h
h_min = min(h)
h_max = max(h)

# Minimum and maximum element size
print("hmin = %g, hmax = %g" % (h_min, h_max))
if rank == 0:
    logfile.write("\n\nMesh: hmin = %g, hmax = %g\n" % (h_min, h_max))

# Shift mesh such that the center is at the origin
print("**** Shifting mesh")
for x in mesh.geometry.x:
    x[0] -= 0.5*height*aspect
    x[1] -= 0.5*height

    # Shift elements at side boundaries to avoid roundoff errors and
    # ensuing problems with periodic boundary mapping when mesh is
    # created with gmsh

    # Distance away from boundary for this to be applied
    margin    = 1.0e-4

    # The larger this number, the more digits are included
    precision = 1.0e6
    if (x[0] < 0.5*height*aspect+margin) or (x[0] > 0.5*height*aspect-margin):
        if x[0] > 0:
            x[0] = int(x[0] * precision + 0.5) / precision
        else:
            x[0] = int(x[0] * precision - 0.5) / precision
        if x[1] > 0:
            x[1] = int(x[1] * precision + 0.5) / precision
        else:
            x[1] = int(x[1] * precision - 0.5) / precision

# Map slave entity to master entity
def periodic_relation(x):
    y = np.zeros_like(x)
    y[0] = - x[0]
    y[1] = x[1]
    y[2] = x[2]
    return y


# ======================================================================
# Function spaces
# ======================================================================

# Finite elements
P2 = VectorElement("Lagrange", mesh.ufl_cell(), degree)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), degree-1)

# Define function spaces
print("**** Defining function spaces")
# Velocity
V = FunctionSpace(mesh, P2)

# Pressure
Q = FunctionSpace(mesh, P1)

# Porosity
X = FunctionSpace(mesh, P1)

# Create mixed function space
if cylinder_mesh:
    # JR -- real element not currently supported by dolfinx
    RE = FiniteElement("Real", mesh.ufl_cell(), 0)
    # Lagrange multiplier for torque
    L = FunctionSpace(mesh, RE)
    print("**** Creating mixed function space")
    TH = MixedElement([P2, P1, RE])
    W = FunctionSpace(mesh, TH)
else:
    TH = MixedElement([P2, P1])
    W = FunctionSpace(mesh, TH)

# Function spaces for h5 output only (don't want constrained_domain option,
# which creates problems in the postprocessing file where BC are not defined)
print("**** Defining h5 output function spaces")
# Velocity
Y = FunctionSpace(mesh, P2)

# Pressure and porosity
Z = FunctionSpace(mesh, P1)

# ======================================================================
# Boundaries and boundary conditions
# ======================================================================

# Define and mark boundaries
print("**** Defining boundaries . . . ")

def on_bottom(x):
    return np.isclose(x[1], -0.5*height)
def on_top(x):
    return np.isclose(x[1], 0.5*height)
def on_left(x):
    return np.isclose(x[0], -0.5*height*aspect)
def on_right(x):
    return np.isclose(x[0], 0.5*height*aspect)

bottom_facets = locate_entities_boundary(mesh, fdim, on_bottom)
top_facets = locate_entities_boundary(mesh, fdim, on_top)
left_facets = locate_entities_boundary(mesh, fdim, on_left)
right_facets = locate_entities_boundary(mesh, fdim, on_right)
facets = np.hstack([bottom_facets, top_facets, left_facets, right_facets])

# Mark mesh
bottom_marks = np.full_like(bottom_facets, 1)
top_marks = np.full_like(top_facets, 2)
left_marks = np.full_like(left_facets, 4)  # miss 3 as used for cylinder
right_marks = np.full_like(right_facets, 5)
marks = np.hstack([bottom_marks, top_marks, left_marks, right_marks])

# Any boundary that is not on the outside edge of the domain is assumed
# to be at the cylinder
if cylinder_mesh:
    def on_cylinder(x):
        eps = 1e-7
        return np.all([x[1] < (0.5 * height - eps),
                  x[1] > (-0.5 * height + eps),
                  x[0] < (0.5 * height * aspect - eps),
                  x[0] > (-0.5 * height * aspect + eps)], axis = 0)
    cylinder_facets = locate_entities_boundary(mesh, fdim, on_cylinder)
    cylinder_marks = np.full_like(cylinder_facets, 3)
    facets = np.hstack([facets, cylinder_facets])
    marks = np.hstack([marks, cylinder_marks])

sort_order = np.argsort(facets)
facet_tags = meshtags(mesh, mesh.topology.dim - 1, facets[sort_order], marks[sort_order])

# Create boundary condition functions
print("**** Setting boundary conditions . . . ")

# Define background velocity field due to the simple shear. This is
# later used to determine velocity perturbations in solid and fluid.
def v_background(x):
    return np.stack((x[1,:], np.zeros(x.shape[1])))    
base_velocity = Function(V)
base_velocity.interpolate(v_background)

# Pinpoint for pressure
def at_pin_point(x):
    return np.isclose(x.T, [0.0, -0.5*height, 0.0])
zero = Function(Q)
with zero.vector.localForm() as zero_local:
    zero_local.set(0.0)

if cylinder_mesh:
    # bit of surface around cylinder
    dss = ds(subdomain_data=facet_tags)
    ds_cylinder = dss(3)

# Create boundary conditions

# specified velocity on top
top_v_dofs = locate_dofs_topological((W.sub(0), V), fdim, top_facets)
Vbc0 = dirichletbc(base_velocity, top_v_dofs, W.sub(0))

# specified velocity on bottom
bottom_v_dofs = locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
Vbc1 = dirichletbc(base_velocity, bottom_v_dofs, W.sub(0))

# set p = 0 at origin
pin_dofs = locate_dofs_geometrical((W.sub(1),Q), at_pin_point)
Vbc2 = dirichletbc(zero, pin_dofs, W.sub(1))

# Collect boundary conditions
Vbcs = [Vbc0, Vbc1, Vbc2]

# Periodic bcs for Stokes
mpc_stokes = MultiPointConstraint(W)
mpc_stokes.create_periodic_constraint_topological(W.sub(0).sub(0), facet_tags, 5, periodic_relation, Vbcs)
mpc_stokes.create_periodic_constraint_topological(W.sub(0).sub(1), facet_tags, 5, periodic_relation, Vbcs)
mpc_stokes.create_periodic_constraint_topological(W.sub(1), facet_tags, 5, periodic_relation, Vbcs) 
mpc_stokes.finalize()

# Periodic bcs for porosity
mpc_porosity = MultiPointConstraint(X)
mpc_porosity.create_periodic_constraint_topological(X, facet_tags, 5, periodic_relation, [])
mpc_porosity.finalize()

# ======================================================================
# Solution functions
# ======================================================================

print("**** Defining solution functions . . .")

# Stokes-like solution (u, p)
U = Function(W)
if cylinder_mesh:
    u, p, omega = split(U)
else:
    u, p = split(U)

# Porosity at time t_n
phi0 = Function(mpc_porosity.function_space)

# ======================================================================
#  Stokes-like and porosity weak formulations
# ======================================================================

# Time step
dt = Constant(mesh, 0.0)

# Get forms
print("Getting porosity form")
a_phi, L_phi = porosity_forms(X, phi0, u, dt)
print("Getting Stokes form")
F_stokes = stokes_forms(W, phi0, dt, param, cylinder_mesh)

# ======================================================================
#  Initial porosity
# ======================================================================

# Set initial porosity field
print("**** Defining initial porosity field ...")

# Interpolate initial porosity
phi_init = physics.initial_porosity(param, X)
phi0.interpolate(phi_init)
#initial_porosity_out.write_mesh(mesh)
initial_porosity_out.write_function(phi0)
initial_porosity_out.close()

# Compute initial mean porosity
if cylinder_mesh:
    mean_phi = assemble_scalar(form(phi0*dx))/(aspect*height*height - math.pi*radius**2)
else:
    mean_phi = assemble_scalar(form(phi0*dx))/(aspect*height*height)
print("**** Mean porosity = %g" % (mean_phi))


# Background velocity field
v0 = Function(V)
v0.interpolate(v_background)
# Interpolate initial condition
U.sub(0).interpolate(v_background)
U.sub(1).interpolate(zero)

# Initial velocity field
# Solve nonlinear Stokes-type system
print("Solving initial Stokes field")
stokes_problem = NonlinearMPCProblem(F_stokes, U, mpc_stokes, bcs = Vbcs)
stokes_solver = NewtonSolverMPC(comm, stokes_problem, mpc_stokes)
stokes_solver.report = True
stokes_solver.rtol = 1e-3
stokes_solver.solve(U)
print("Finished solving initial Stokes field")

# Calculate the torque and deviation from rigid body rotation - both
# should be zero
if cylinder_mesh:
    physics.print_cylinder_diagnostics(phi0, p, u, omega, param, \
                                       mesh, ds_cylinder, logfile)

# Write data to files for quick visualisation
srate      = physics.strain_rate(u)
shear_visc = physics.eta(phi0, srate, param)
bulk_visc  = physics.zeta(phi0, shear_visc, param)
perm       = physics.perm(phi0, param)
core.write_vtk(0.0, V, Q, phi0, U, v0, shear_visc, bulk_visc, perm, srate, \
                   vel_pert_out, velocity_out, pressure_out, \
                   porosity_out, divU_out, shear_visc_out, \
                   bulk_visc_out, perm_out, strain_rate_out)

## Write data to h5 files for postprocessing
## Porosity
#phi = Function(Z)
#phi.interpolate(phi0)
##h5file_phi.write(phi, "porosity_%d" % 0)
##h5file_phi.write(mesh, "mesh_file")

## Velocity
#vel = Function(Y)
#vel.interpolate(project(u))
##h5file_vel.write(vel, "velocity_%d" % 0)
##h5file_vel.write(mesh, "mesh_file")

## Pressure
#pres = Function(Z)
#pres.interpolate(project(p))
##h5file_pres.write(pres, "pressure_%d" % 0)
##h5file_pres.write(mesh, "mesh_file")

if rank == 0:
    logfile.write("\nTime step 0: t = 0\n")

# Compare results to shear band growth rates from plane wave
# benchmark.
t = 0.0
#comm.Barrier()
#if rank == 0 and initial_porosity_field == 'plane_wave':
#    analysis.plane_wave_analysis(Q, u, t, param, logfile)
#comm.Barrier()

# Computation of analytical compaction rates around cylinder.
#if initial_porosity_field == 'uniform' and cylinder_mesh:
#    analysis.compaction_cylinder_analysis(Q, V, u, p, shear_visc, \
#                                              bulk_visc, param, logfile)

# ======================================================================
#  Compute initial time step
# ======================================================================

# Set time step
dt.value = core.compute_dt(U, cfl, h_min, cylinder_mesh)

print("initial dt = %g \n" % dt.value)
print("-------------------------------\n")

# ======================================================================
#  Time loop
# ======================================================================

## Create a direct linear solver for porosity
sol_opts = {"ksp_type": "preonly", "pc_type": "lu"}  # use LU decomposition
problem_phi = LinearProblem(a_phi, L_phi, mpc_porosity, u = phi0, petsc_options=sol_opts)
    
tcount = 1
while (t < tmax):
    if t < tmax and t + dt.value > tmax:
        dt.value = tmax - t

    print("Time step %d: time slab t = %g to t = %g" % (tcount, t, t + dt.value))
    if rank == 0:
        logfile.write("\nTime step %d: t = %g\n" % (tcount, t + dt.value))

    # Solve for U_n+1 and phi_n+1
    # Compute U and phi1, and update phi0 <- phi1
    print("**** t = %g: Solve phi and U" % t)

    # Solve linear porosity advection system
    problem_phi.solve()
    
    print("Phi vector norms: %g" \
             % (phi0.vector.norm(Norm.l2))) #, b_phi.norm(Norm.l2)))
    if rank == 0:
        logfile.write("Phi vector norms: %g\n" \
                          % (phi0.vector.norm(Norm.l2) ))#, b_phi.norm("l2")))

    # Solve linear Stokes-type system
    print("Solving Stokes problem")
    stokes_solver.solve(U)

    # Calculate the torque and deviation from rigid body rotation -
    # both should be zero
    if cylinder_mesh:
        physics.print_cylinder_diagnostics(phi0, p, u, omega, \
                                           param, mesh, ds_cylinder, \
                                           logfile)

    # Compare results to shear band growth rates from plane wave
    # benchmark
    #comm.Barrier()
    #if rank == 0 and initial_porosity_field == 'plane_wave':
    #    analysis.plane_wave_analysis(Q, u, t, param, logfile)
    #comm.Barrier()

    # Write results to files with output frequency
    if tcount % out_freq == 0:
        print("Writing output")
        # Write data to files for quick visualisation
        srate = physics.strain_rate(u)
        core.write_vtk(t, V, Q, phi0, U, v0, shear_visc, bulk_visc, perm, srate, \
                           vel_pert_out, velocity_out, pressure_out, \
                           porosity_out, divU_out, shear_visc_out, \
                           bulk_visc_out, perm_out, strain_rate_out)

        ## Write data to h5 files for postprocessing
        ## Porosity
        #phi.interpolate(phi1)
        #h5file_phi.write(phi, "porosity_%d" % tcount)
        ## Velocity
        #vel.interpolate(project(u))
        #h5file_vel.write(vel, "velocity_%d" % tcount)
        ## Pressure
        #pres.interpolate(project(p))   
        #h5file_pres.write(pres, "pressure_%d" % tcount)

    # Check that phi is within allowable bounds
    # JR - turn off for now
    #physics.check_phi(phi1, logfile)

    # Compute new time step
    dt.value = core.compute_dt(U, cfl, h_min, cylinder_mesh)

    print("**** New time step dt = %g\n" % dt.value)
    print("-------------------------------\n")

    t      += dt.value
    tcount += 1

if rank == 0:
    logfile.write("\nEOF\n")
    logfile.close()
