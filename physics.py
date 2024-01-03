#!/usr/bin/env python3

# ======================================================================
# physics.py
#
# Contains a variety of viscosity formulations and initial porosity fields.
#
# Authors:
# Laura Alisic, University of Cambridge
# Sander Rhebergen, University of Oxford
# Garth Wells, University of Cambridge
# John Rudge, University of Cambridge
#
# Last modified: 26 Jan 2015 by Laura Alisic
# ======================================================================

from mpi4py import MPI
# syntax change: from dolfin import info, UserExpression, 
from ufl import sqrt, inner, FacetNormal, dot, sym, grad, exp
from dolfinx.fem import Expression, assemble, FunctionSpace
import numpy, math, sys

comm = MPI.COMM_WORLD

# ======================================================================

def eta(phi, srate, param):
    """Porosity dependent viscosity:
       eta = exp(-alpha * (phi - phiB))"""

    alpha = param['alpha']
    phiB  = param['phiB']
    n     = param['stress_exp']

    visc = exp(-alpha * (phi - phiB)) * srate**((1.0-n)/n)

    return visc

# ======================================================================

def testing_eta(phi, param):
    """Porosity dependent viscosity with alternative scaling:
       eta = B0 * exp(-alpha * phiB * (phi - 1.0))"""

    alpha = param['alpha']
    phiB  = param['phiB']
    rzeta = param['rzeta']

    B0    = 1.0 / (rzeta + (4.0/3.0)) # eta_0/(xi0+4eta0/3)
    visc  = B0 * exp(-alpha * phiB * (phi - 1.0))

    return visc

# ======================================================================

def zeta(phi, eta, param):
    """Bulk viscosity scaled to shear viscosity:
       zeta = eta * phi**-m"""

    m    = param['m']
    phiB = param['phiB']

    visc = eta * (phi / phiB)**(-m)

    return visc

# ======================================================================

def perm(phi, param):
    """Porosity-dependent permeability:
       perm = phi**n"""

    n    = param['n']
    phiB = param['phiB']

    perm = (phi / phiB)**n

    return perm

# ======================================================================

def strain_rate(u):
    """Second invariant of the strain rate tensor"""

    # Strain rate tensor
    srate = sym(grad(u))

    # Second invariant
    sec_inv = sqrt(0.5*inner(srate, srate))

    return sec_inv

# ======================================================================

def stress(phi, p, u, param):
    """Total stress tensor"""

    srate        = strain_rate(u)
    shear_visc   = eta(phi, srate, param)
    bulk_visc    = zeta(phi, shear_visc, param)
    rzeta        = param['rzeta']

    return 2.0*shear_visc*sym(grad(u)) + (rzeta*bulk_visc - \
           (2.0/3.0)*shear_visc)*div(u)*Identity(2) \
           - p*Identity(2)

# ======================================================================

def traction(phi, p, u, param, mesh):
    """Traction vector"""

    sigma = stress(phi, p, u, param)
    n     = FacetNormal(mesh)

    return dot(sigma, n)

# ======================================================================

def force(phi, p, u, param, mesh, ds_object):
    """Force on an object"""

    t = traction(phi, p, u, param, mesh)

    return t*ds_object         # How would we go about assembling this?

# ======================================================================

def torque(phi, p, u, param, mesh, ds_object):
    """Torque on an object"""

    t       = traction(phi, p, u, param, mesh)
    x_perp  = Expression(("-x[1]", "x[0]"), degree = 1)

    return dot(x_perp, t)*ds_object

# ======================================================================

def calculate_rigid_rotation_error(u, omega, ds_object):
    """Calculate rigid body rotation error of the cylinder"""

    x_perp     = Expression(("-x[1]", "x[0]"), degree = 1)
    u_rotation = omega * x_perp
    error      = ((u_rotation - u)**2)*ds_object

    return assemble(error)

# ======================================================================

def get_rotation_rate(omega, mesh):
    """Get the rotation rate of the cylinder"""
    #TODO: There must be a better way of doing this!!

    L = FunctionSpace(mesh, "Real", 0)
    rot_rate_obj = Function(L)
    rot_rate_obj.assign(omega)
    return rot_rate_obj.vector().array()

# ======================================================================

def print_cylinder_diagnostics(phi, p, u, omega, param, mesh, ds_cylinder, logfile):
    """Print various diagnostics about the cylinder, like torque, rigid body rotationness, and rotation rate"""

    # MPI
    comm = MPI.comm_world

    total_torque = assemble(torque(phi, p, u, param, mesh, ds_cylinder))
    info("**** Torque in numerical solution = %g" % total_torque)
    if MPI.rank(comm) == 0:
        logfile.write("Torque in numerical solution = %g\n" % total_torque)

    total_rigid_rotation_error = calculate_rigid_rotation_error(u, omega, ds_cylinder)
    info("**** Rigid body rotation error in numerical solution = %g" % total_rigid_rotation_error)
    if MPI.rank(comm) == 0:
        logfile.write("Rigid body rotation error in numerical solution = %g\n" % total_rigid_rotation_error)

    # FIXME: This gives an error about Function Spaces not being the same at the moment
    # (probably need the same constrained pbc space as in the main code?)
    #rotation = get_rotation_rate(omega, mesh)
    #MPI.barrier(comm)
    #if MPI.rank(comm) == 0:
    #    info("**** Rotation rate of cylinder = %g" % rotation)
    #    logfile.write("Rotation rate of cylinder = %g\n" % rotation)
    #MPI.barrier(comm)

# ======================================================================

def initial_porosity(param, X):
    """Initial condition for porosity"""

    initial_porosity_field = param['initial_porosity_field']
    read_initial_porosity  = param['read_initial_porosity']
    phiB                   = param['phiB']
    amplitude              = param['amplitude']
    k_0                    = math.pi * param['k_0']
    angle_0                = math.pi * param['angle_0']
    nr_sines               = param['nr_sines']
    meshtype               = param['meshtype']
    aspect                 = param['aspect']
    el                     = param['el']
    height                 = param['height']

    phi_min = phiB - amplitude
    phi_max = phiB + amplitude

    # MPI command needed for HDF5
    comm = MPI.comm_world

    if (read_initial_porosity == 1):

        initial_porosity_in    = param['initial_porosity_in']
        h5file_init = HDF5File(comm, initial_porosity_in, "r")
        phi_input   = Function(X)
        h5file_init.read(phi_input, "initial_porosity")
        phi_scaled  = phiB + (phi_input-0.5)*2.0*amplitude
        phi_out     = project(phi_scaled)

        return phi_out

    # NOTE: FFT filtering of random field only works in serial at the moment
    elif (initial_porosity_field == 'random') and (read_initial_porosity == 0):

        # Create temporary mesh and random porosity numpy array of the same size
        # Create larger mesh so that interpolation at the edges does not leave artefacts
        height_new = height * 1.1
        mesh       = RectangleMesh(Point(-0.5*aspect*height_new, -0.5*height_new), Point(aspect*height_new, \
                                   height_new), int(aspect*el), int(el), meshtype)
        elements   = int(sqrt(mesh.num_vertices()))
        phi_array  = 2.0 * amplitude * numpy.random.rand(elements, elements) + phi_min

        # Filter mesh with 2D FFT to get rid of highest frequency content;
        # fftshift moves zero-frequency component to the center of the array
        phi_freq          = numpy.fft.fftshift(numpy.fft.fft2(phi_array)) # from spatial to frequency domain
        phi_freq_filtered = numpy.zeros(phi_freq.shape, dtype = complex)  # filtered spectrum

        # Define filter spectrum
        filter_range = 1.0 # fraction of spectrum maintained
        width        = int((filter_range * min(phi_array.shape)) / 2)
        centre_x     = phi_freq.shape[0]/2 # spectrum centre in x dimension
        centre_y     = phi_freq.shape[1]/2 # spectrum centre in y dimension

        # Only store the coefficients for ifft that are within the desired part of the spectrum
        # Low-pass filter: cut the high-frequency ends of the shifted spectrum
        for i in range(int(centre_x - width), int(centre_x + width)):
            for j in range(int(centre_y - width), int(centre_y + width)):
                phi_freq_filtered[i,j] = phi_freq[i,j]

        # From frequency to spatial domain
        phi_spatial = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(phi_freq)))

        # Interpolate filtered numpy array onto arbitrary mesh of the same size as the array
        P            = FunctionSpace(mesh, "Lagrange", 1)
        phi_filtered = Function(P)
        phi_filtered.vector()[:] = phi_spatial.flatten()

        return phi_filtered

    else:
        if (initial_porosity_field == 'perlin'):
            from noise import pnoise2

        elif (initial_porosity_field == 'sinusoidal'):
            shift    = numpy.random.rand(2, nr_sines)
            k_rand   = k_0 * numpy.random.rand(2, nr_sines)

        class IC_porosity(UserExpression):
            def eval(self, v, x):

                # Uniform initial field
                if (initial_porosity_field == 'uniform'):
                    v[0] = phiB

                # Plane wave initial field
                if (initial_porosity_field == 'plane_wave'):
                    v[0] = phiB*(1.0 + amplitude*cos(k_0*(x[0]*sin(angle_0) + x[1]*cos(angle_0))))

                # Sinusoidal initial field
                # FIXME: Only works properly when run in serial (otherwise see proc boundaries)
                if (initial_porosity_field == 'sinusoidal'):
                    nr       = 0
                    v[0]     = 0.0
                    while (nr < nr_sines):
                        v[0]   += (1.0 / float(nr_sines)) * phiB*(1.0 + amplitude * \
                                   sin(k_rand[0][nr]*(x[0] + shift[0][nr])) * \
                                   sin(k_rand[1][nr]*(x[1] + shift[1][nr])))
                        nr     += 1

                # Perlin noise initial field
                if initial_porosity_field == 'perlin':
                    xn = (x[0]/height) % aspect
                    yn = (x[1]/height) % 1.0
                    v[0] = phiB*(1.0 + amplitude*pnoise2(xn, yn, octaves=16, persistence=0.5, repeatx=aspect, repeaty=1.0))

        return IC_porosity()

# ======================================================================

def check_phi(phi, logfile):
    """Check that porosity phi is within allowable range (0, 1)"""

    phi_min = phi.vector().min()
    phi_max = phi.vector().max()
    info("**** Minimum porosity = %g" % (phi_min))
    info("**** Maximum porosity = %g" % (phi_max))

    if MPI.rank(comm) == 0:
        logfile.write("Minimum porosity = %g; maximum porosity = %g\n" % (phi_min, phi_max))

    if phi_min < 0.0 or phi_max > 1.0:
        info("**** Porosity out of bounds --- EXITING\n")
        if MPI.rank(comm) == 0:
            logfile.write("Porosity out of bounds --- EXITING\n")
            logfile.write("\nEOF\n")
            logfile.close()
        sys.exit()

# ======================================================================

# EOF
