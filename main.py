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
# Last modified: 26 Jan 2015 by Laura Alisic
# ======================================================================

# syntax change: from dolfin import DirichletBC, RectangleMesh, parameters
# syntax change: from dolfin import Matrix, Vector, LUSolver, SubDomain, CompiledSubDomain
# syntax change: from dolfin import near, info, solve,  project
# syntax change: from dolfin import MeshFunction, Mesh, Point
# syntax change: from dolfin import HDF5File, File
from mpi4py import MPI
from dolfinx.fem import Function, Constant, dirichletbc, FunctionSpace,   Expression, assemble
from dolfinx.mesh import create_rectangle
from dolfinx.la import MatrixCSR, Vector
from ufl import sqrt, inner, sym, dot, div, dx, grad, TrialFunction, TestFunction,  TestFunctions, CellDiameter, lhs, rhs, split, VectorElement, FiniteElement, MixedElement
import numpy, sys, math
import core
import physics
import analysis
import mesh_gen
import mesh_gen_uniform
import datetime

#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["optimize"] = True

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
    f  = Constant((0.0, 0.0))
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

    #return lhs(F), rhs(F)
    return F

# Write output to process 0 only
#parameters["std_out_all_processes"] = False

# Needed for interpolating fields without throwing an error
#parameters['allow_extrapolation'] = True

# Log level
#set_log_level(DEBUG)

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

# Output files for quick visualisation
output_dir     = "output/"
extension      = "pvd"   # "xdmf" or "pvd"
initial_porosity_out = File(output_dir + "initial_porosity." + extension)
velocity_out   = File(output_dir + "velocity." + extension)
vel_pert_out   = File(output_dir + "velocity_perturbations." + extension)
pressure_out   = File(output_dir + "pressure." + extension)
porosity_out   = File(output_dir + "porosity." + extension)
shear_visc_out = File(output_dir + "shear_viscosity." + extension)
bulk_visc_out  = File(output_dir + "bulk_viscosity." + extension)
perm_out       = File(output_dir + "permeability." + extension)
compaction_out = File(output_dir + "compaction." + extension)
num_ds_dt_out  = File(output_dir + "ds_dt." + extension)
divU_out       = File(output_dir + "div_u." + extension)
strain_rate_out = File(output_dir + "strain_rate." + extension)

# MPI command needed for HDF5
comm = MPI.comm_world

# Output files for further postprocessing
h5file_phi     = HDF5File(comm, "porosity.h5", "w")
h5file_vel     = HDF5File(comm, "velocity.h5", "w")
h5file_pres    = HDF5File(comm, "pressure.h5", "w")

# Initialise logfile
logfile = open(logname, "w", encoding="utf-8")
if MPI.rank(comm) == 0:
    logfile.write(str(datetime.datetime.now()))

# Print params to logfile
if MPI.rank(comm) == 0:
    logfile.write("\n\nRun parameters:\n")
    for item in sorted(list(param.keys()), key=str.lower):
        logfile.write("%s = %s\n" % (item, param[item]))

# ======================================================================
# Mesh
# ======================================================================

# Create mesh
if read_mesh:
    print("**** Reading mesh file: %s" % meshfile)
    mesh = Mesh(meshfile)
else:
    print("**** Generating mesh . . . ")
    if cylinder_mesh:

        # Create a mesh with gmsh
        # NOTE: running this in parallel throws an error about lifeline lost.
        MPI.barrier(comm)
        #mesh_gen_uniform.cylinder_mesh_gen(filename=meshfile, \
        mesh_gen.cylinder_mesh_gen(filename=meshfile, \
                                   aspect=aspect, \
                                   N=el, \
                                   h=height, \
                                   rel_radius=(radius/height))
        MPI.barrier(comm)
        mesh = Mesh(meshfile)
    else:
        mesh = RectangleMesh(Point(0, 0), Point(aspect*height, height), \
                                 int(aspect*el), int(el), meshtype)

# Smallest element size. Used to determine time step
h_min = MPI.min(comm, mesh.hmin())
h_max = MPI.max(comm, mesh.hmax())

# Minimum and maximum element size
print("hmin = %g, hmax = %g" % (h_min, h_max))
if MPI.rank(comm) == 0:
    logfile.write("\n\nMesh: hmin = %g, hmax = %g\n" % (h_min, h_max))

# Shift mesh such that the center is at the origin
print("**** Shifting mesh")
for x in mesh.coordinates():
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

# Subdomain for periodic boundary condition (left and right boundaries)
class PeriodicBoundary(SubDomain):
    """Define periodic boundaries"""
    def __init__(self, tol):
        SubDomain.__init__(self, tol)

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],-0.5*height*aspect, 1.0e-11)

    def map(self, x, y):
        """Map slave entity to master entity"""
        y[0] = x[0] - height*aspect
        y[1] = x[1]

# Create an object to prevect director going out of scope. Might fix
# later.
pbc = PeriodicBoundary(1.0e-6)

#mf = PeriodicBoundaryComputation.masters_slaves(mesh, pbc, 1)
#File("periodic_boundaries.xdmf") << mf

# ======================================================================
# Function spaces
# ======================================================================

# Finite elements
P2 = VectorElement("Lagrange", mesh.ufl_cell(), degree)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), degree-1)
RE = FiniteElement("Real", mesh.ufl_cell(), 0)

# Define function spaces
print("**** Defining function spaces")
# Velocity
V = FunctionSpace(mesh, P2, constrained_domain=pbc)

# Pressure
Q = FunctionSpace(mesh, P1, constrained_domain=pbc)

# Porosity
X = FunctionSpace(mesh, P1, constrained_domain=pbc)

# Create mixed function space
if cylinder_mesh:
    # Lagrange multiplier for torque
    #L = FunctionSpace(mesh, "Real", 0, constrained_domain=pbc)
    L = FunctionSpace(mesh, RE)
    print("**** Creating mixed function space")
    TH = MixedElement([P2, P1, RE])
    W = FunctionSpace(mesh, TH, constrained_domain=pbc)
else:
    TH = MixedElement([P2, P1])
    W = FunctionSpace(mesh, TH, constrained_domain=pbc)

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

# Holder for domain markers
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

# Mark bottom boundary as 1
bottom = CompiledSubDomain("x[1] < (-0.5*height + DOLFIN_EPS) && \
                             on_boundary".replace("height", str(height)))
bottom.mark(sub_domains, 1)

# Mark top boundary as 2
top = CompiledSubDomain("x[1] > ( 0.5*height - DOLFIN_EPS) && \
                          on_boundary".replace("height", str(height)))
top.mark(sub_domains, 2)

# Any boundary that is not on the outside edge of the domain is assumed
# to be at the cylinder
if cylinder_mesh:
    cylinder_str = "x[1] < ( 0.5*height - DOLFIN_EPS) && \
                    x[1] > (-0.5*height + DOLFIN_EPS) && \
                    x[0] < ( 0.5*height*aspect - DOLFIN_EPS) && \
                    x[0] > (-0.5*height*aspect + DOLFIN_EPS) && \
                    on_boundary".replace("height", str(height)).replace("aspect", str(aspect))

    cylinder = CompiledSubDomain(cylinder_str)
    cylinder.mark(sub_domains, 3) # mark cylinder boundary as 3

# Create boundary condition functions
print("**** Setting boundary conditions . . . ")

# vector for velocity on top boundary
topv  = Expression((" 0.5*h",  "0.0"), h = height, degree=1)

# vector for velocity on bottom boundary
bottomv   = Expression(("-0.5*h", "0.0"), h = height , degree=1)

# Pinpoint for pressure
pin_str = "std::abs(x[0]) < DOLFIN_EPS && std::abs(x[1]) < (-0.5 + DOLFIN_EPS)"
pinpoint = CompiledSubDomain(pin_str)

if cylinder_mesh:
    # bit of surface around cylinder
    dss = ds(subdomain_data=sub_domains)
    ds_cylinder = dss(3)

# Create boundary conditions

# specified velocity on top
Vbc0 = DirichletBC(W.sub(0), topv, top)

# specified velocity on bottom
Vbc1 = DirichletBC(W.sub(0), bottomv, bottom)

# set p = 0 at origin
Vbc2 = DirichletBC(W.sub(1), 0.0, pinpoint, "pointwise")

# Collect boundary conditions
Vbcs = [Vbc0, Vbc1, Vbc2]

# ======================================================================
# Solution functions
# ======================================================================

info ("**** Defining solution functions . . .")

# Stokes-like solution (u, p)
U = Function(W)
if cylinder_mesh:
    u, p, omega = split(U)
else:
    u, p = split(U)

# Porosity at time t_n
phi0 = Function(X)

# Porosity at time t_n+1
phi1 = Function(X)

# ======================================================================
#  Stokes-like and porosity weak formulations
# ======================================================================

# Time step. Use Expression to avoid form re-compilation
dt = Expression("dt", dt=0.0, degree = 1)

# Get forms
print("Getting porosity form")
a_phi, L_phi = porosity_forms(X, phi0, u, dt)
print("Getting Stokes form")
#a_stokes, L_stokes = stokes_forms(W, phi0, dt, param, cylinder_mesh)
F_stokes = stokes_forms(W, phi0, dt, param, cylinder_mesh)
#a_stokes = lhs(F_stokes)
#L_stokes = rhs(F_stokes)

# ======================================================================
#  Initial porosity
# ======================================================================

# Set initial porosity field
print("**** Defining initial porosity field ...")

# Interpolate initial porosity
phi_init = physics.initial_porosity(param, X)
phi0.interpolate(phi_init)
initial_porosity_out << phi0

# Compute initial mean porosity
if cylinder_mesh:
    mean_phi = assemble(phi0*dx)/(aspect*height*height - math.pi*radius**2)
else:
    mean_phi = assemble(phi0*dx)/(aspect*height*height)
print("**** Mean porosity = %g" % (mean_phi))

# Define background velocity field due to the simple shear. This is
# later used to determine velocity perturbations in solid and fluid.
v_background = Expression(("x[1]", "0.0"), degree = 1)

# Background velocity field
v0 = Function(V)
v0.interpolate(v_background)

# Initial velocity field
# Solve nonlinear Stokes-type system
print("Solving initial Stokes field")
# FIXME: Solve fails with the error 'All terms in form must have same rank.'
#solve(a_stokes == L_stokes, U, Vbcs, form_compiler_parameters={"quadrature_degree": 3, "optimize": True} )
solve(F_stokes == 0, U, Vbcs, form_compiler_parameters={"quadrature_degree": 3, "optimize": True}, \
                              solver_parameters={"newton_solver": {"relative_tolerance": 1e-3}} )
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
core.write_vtk(Q, p, phi0, u, v0, shear_visc, bulk_visc, perm, srate, \
                   vel_pert_out, velocity_out, pressure_out, \
                   porosity_out, divU_out, shear_visc_out, \
                   bulk_visc_out, perm_out, strain_rate_out)

# Write data to h5 files for postprocessing
# Porosity
phi = Function(Z)
phi.interpolate(phi0)
h5file_phi.write(phi, "porosity_%d" % 0)
h5file_phi.write(mesh, "mesh_file")

# Velocity
vel = Function(Y)
vel.interpolate(project(u))
h5file_vel.write(vel, "velocity_%d" % 0)
h5file_vel.write(mesh, "mesh_file")

# Pressure
pres = Function(Z)
pres.interpolate(project(p))
h5file_pres.write(pres, "pressure_%d" % 0)
h5file_pres.write(mesh, "mesh_file")

if MPI.rank(comm) == 0:
    logfile.write("\nTime step 0: t = 0\n")

# Compare results to shear band growth rates from plane wave
# benchmark.
t = 0.0
MPI.barrier(comm)
if MPI.rank(comm) == 0 and initial_porosity_field == 'plane_wave':
    analysis.plane_wave_analysis(Q, u, t, param, logfile)
MPI.barrier(comm)

# Computation of analytical compaction rates around cylinder.
#if initial_porosity_field == 'uniform' and cylinder_mesh:
#    analysis.compaction_cylinder_analysis(Q, V, u, p, shear_visc, \
#                                              bulk_visc, param, logfile)

# ======================================================================
#  Compute initial time step
# ======================================================================

# Set time step
dt.dt = core.compute_dt(U, cfl, h_min, cylinder_mesh)

print("initial dt = %g \n" % dt.dt)
print("-------------------------------\n")

# ======================================================================
#  Time loop
# ======================================================================

# Solver matrices
A_phi, A_stokes = Matrix(), Matrix()

# Solver RHS
b_phi, b_stokes = Vector(), Vector()

# Sparsity pattern reset
reset_sparsity_flag = True

# Create a direct linear solver for porosity
solver_phi = LUSolver(A_phi)

# FIXME: Check matrices for symmetry
# Create a conjugate gradient linear solver for porosity
#solver_phi = KrylovSolver("cg", "none")
#solver_phi.set_operator(A_phi)

# Create linear solver for Stokes-like problem
solver_U = LUSolver(A_stokes)

tcount = 1
while (t < tmax):
    if t < tmax and t + dt.dt > tmax:
        dt.dt = tmax - t

    print("Time step %d: time slab t = %g to t = %g" % (tcount, t, t + dt.dt))
    if MPI.rank(comm) == 0:
        logfile.write("\nTime step %d: t = %g\n" % (tcount, t + dt.dt))

    # Solve for U_n+1 and phi_n+1
    # Compute U and phi1, and update phi0 <- phi1
    print("**** t = %g: Solve phi and U" % t)

    # Assemble system for porosity advection
    ffc_parameters = dict(quadrature_degree=3, optimize=True)
    #assemble(a_phi, tensor=A_phi, reset_sparsity=reset_sparsity_flag, \
    assemble(a_phi, tensor=A_phi, \
                 form_compiler_parameters=ffc_parameters)
    #assemble(L_phi, tensor=b_phi, reset_sparsity=reset_sparsity_flag, \
    assemble(L_phi, tensor=b_phi, \
                 form_compiler_parameters=ffc_parameters)

    # Solve linear porosity advection system
    solver_phi.solve(phi1.vector(), b_phi)
    print("Phi vector norms: %g, %g" \
             % (phi1.vector().norm("l2"), b_phi.norm("l2")))
    if MPI.rank(comm) == 0:
        logfile.write("Phi vector norms: %g, %g\n" \
                          % (phi1.vector().norm("l2"), b_phi.norm("l2")))

    # Update porosity
    phi0.assign(phi1)

    # FIXME: The Stokes solve will currently break (need nonlinear solver)

    # Assemble Stokes-type system
    #ffc_parameters = dict(quadrature_degree=3, optimize=True)
    # TODO: Start non-Newtonian loop here, need re-assembling for each iteration?
    ##assemble(a_stokes, tensor=A_stokes, reset_sparsity=reset_sparsity_flag, \
    #assemble(a_stokes, tensor=A_stokes, \
    #              form_compiler_parameters=ffc_parameters)
    ##assemble(L_stokes, tensor=b_stokes, reset_sparsity=reset_sparsity_flag, \
    #assemble(L_stokes, tensor=b_stokes, \
    #             form_compiler_parameters=ffc_parameters)
    #for bc in Vbcs:
    #    bc.apply(A_stokes, b_stokes)

    # Solve linear Stokes-type system
    print("Solving Stokes problem")
    #solver_U.solve(U.vector(), b_stokes)
    #print("U vector norms: %g, %g" % (U.vector().norm("l2"), b_stokes.norm("l2")))
    #if MPI.rank(comm) == 0:
    #    logfile.write("U vector norms: %g, %g\n" \
    #                      % (U.vector().norm("l2"), b_stokes.norm("l2")))
    solve(F_stokes == 0, U, Vbcs, form_compiler_parameters={"quadrature_degree": 3, "optimize": True}, \
                                  solver_parameters={"newton_solver": {"relative_tolerance": 1e-3}} )

    # Prevent sparsity being re-computed at next solve
    reset_sparsity_flag = False

    # Calculate the torque and deviation from rigid body rotation -
    # both should be zero
    if cylinder_mesh:
        physics.print_cylinder_diagnostics(phi1, p, u, omega, \
                                           param, mesh, ds_cylinder, \
                                           logfile)

    # Compare results to shear band growth rates from plane wave
    # benchmark
    MPI.barrier(comm)
    if MPI.rank(comm) == 0 and initial_porosity_field == 'plane_wave':
        analysis.plane_wave_analysis(Q, u, t, param, logfile)
    MPI.barrier(comm)

    # Write results to files with output frequency
    if tcount % out_freq == 0:
        # Write data to files for quick visualisation
        srate = physics.strain_rate(u)
        core.write_vtk(Q, p, phi1, u, v0, shear_visc, bulk_visc, perm, srate, \
                           vel_pert_out, velocity_out, pressure_out, \
                           porosity_out, divU_out, shear_visc_out, \
                           bulk_visc_out, perm_out, strain_rate_out)

        # Write data to h5 files for postprocessing
        # Porosity
        phi.interpolate(phi1)
        h5file_phi.write(phi, "porosity_%d" % tcount)
        # Velocity
        vel.interpolate(project(u))
        h5file_vel.write(vel, "velocity_%d" % tcount)
        # Pressure
        pres.interpolate(project(p))   
        h5file_pres.write(pres, "pressure_%d" % tcount)

    # Check that phi is within allowable bounds
    physics.check_phi(phi1, logfile)

    # FIXME: Should we add dolfin::Function::sub_vector() function to
    #        avoid the deep copies? Sub-vector views are supported by
    #        PETSc and EpetraExt

    # Compute new time step
    dt.dt = core.compute_dt(U, cfl, h_min, cylinder_mesh)

    print("**** New time step dt = %g\n" % dt.dt)
    print("-------------------------------\n")

    t      += dt.dt
    tcount += 1

if MPI.rank(comm) == 0:
    logfile.write("\nEOF\n")
    logfile.close()
