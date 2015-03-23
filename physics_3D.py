#!/usr/bin/env python

# ======================================================================
# physics_3D.py
#
# Contains function to create 3D initial porosity fields, only tested 
# for the initial random field.
#
# Authors:
# Laura Alisic, University of Cambridge
# Sander Rhebergen, University of Oxford
# Garth Wells, University of Cambridge
# John Rudge, University of Cambridge
#
# Last modified: 21 Jan 2015 by Laura Alisic
# ======================================================================

from dolfin import *
import numpy, math, sys

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

    if (read_initial_porosity == 1):

        # MPI command needed for HDF5
        comm = mpi_comm_world()

        initial_porosity_in    = param['initial_porosity_in']
        h5file_init = HDF5File(comm, initial_porosity_in, "r")
        phi_input   = Function(X)
        h5file_init.read(phi_input, "initial_porosity", False)
        phi_scaled  = phiB + (phi_input-0.5)*2.0*amplitude
        phi_out     = project(phi_scaled, X)

        return phi_out

    # NOTE: FFT filtering of random field only works in serial at the moment
    elif (initial_porosity_field == 'random') and (read_initial_porosity == 0):

        # Create temporary mesh and random porosity numpy array of the same size
        # Create larger mesh so that interpolation at the edges does not leave artefacts
        el = el+10
        height_new = height * 1.05
        mesh       = BoxMesh(-0.5*aspect*height_new, -0.5*aspect*height_new, -0.5*height_new, \
                             aspect*height_new, aspect*height_new, height_new, \
                             int(aspect*el), int(aspect*el), int(el))
        phi_array  = 2.0 * amplitude * numpy.random.rand(int(aspect*el+1), int(aspect*el+1), int(el+1)) + phi_min

        # Filter mesh with FFT to get rid of highest frequency content;
        # fftshift moves zero-frequency component to the center of the array
        phi_freq          = numpy.fft.fftshift(numpy.fft.fftn(phi_array)) # from spatial to frequency domain
        phi_freq_filtered = numpy.zeros(phi_freq.shape, dtype = complex)  # filtered spectrum

        # Define filter spectrum
        filter_range = 1.0 # fraction of spectrum maintained
        width        = int((filter_range * min(phi_array.shape)) / 2)
        centre_x     = phi_freq.shape[0]/2 # spectrum centre in x dimension
        centre_y     = phi_freq.shape[1]/2 # spectrum centre in y dimension
        centre_z     = phi_freq.shape[2]/2 # spectrum centre in z dimension

        # Only store the coefficients for ifft that are within the desired part of the spectrum
        # Low-pass filter: cut the high-frequency ends of the shifted spectrum
        for i in range(centre_x - width, centre_x + width):
            for j in range(centre_y - width, centre_y + width):
                for k in range(centre_z - width, centre_z + width):
                    phi_freq_filtered[i,j,k] = phi_freq[i,j,k]

        # From frequency to spatial domain
        phi_spatial = numpy.real(numpy.fft.ifftn(numpy.fft.ifftshift(phi_freq)))

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

        class IC_porosity(Expression):
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

# EOF
