#!/usr/bin/env python3

# ======================================================================
# analysis.py
#
# Computes a variety of diagnostics and benchmark quantities.
#
# Authors:
# Laura Alisic, University of Cambridge
# Sander Rhebergen, University of Oxford
# Garth Wells, University of Cambridge
# John Rudge, University of Cambridge
#
# Last modified: 26 Jan 2015 by Laura Alisic
# ======================================================================

# syntax change: from dolfin import info, errornorm, project, norm, File
from dolfinx.fem import Expression, assemble, Constant, Function
from ufl import sin, cos, sqrt, div, dx, exp
import numpy, sys, math
import scipy
from scipy.special import kv, kvp  # Bessel functions and derivatives

# ======================================================================

# Only computed at zero time step
def compaction_cylinder_analysis(Q, V, u, p, shear_visc, bulk_visc, param, logfile):
    """Computation of analytical compaction rate around cylinder"""

    print("**** Computing analytical compaction ...")

    radius  = param['radius']
    R       = param['R']
    rzeta   = param['rzeta']
    aspect  = param['aspect']
    height  = param['height']

    # Viscosity ratio
    B0        = 1.0 / (rzeta + (4.0/3.0))

    # Strain
    gamma     = 1.0

    # Mapping from numerics scaling to analytical scaling
    x_scaling = 1.0 / R
    P_scaling = B0
    v_scaling = 1.0 / R

    # Constants in the analytical solution
    a     = radius * x_scaling
    C_val = - (a**4 * kvp(2, a, 1)) / ((4.0 * B0 * kv(1, a)) - (a**2 * kvp(2, a, 1)))
    D_val = (a**4 / 4.0) + (4.0 * a**3 * B0 * kv(2, a)) / ((4.0 * B0 * kv(1, a)) - \
            (a**2 * kvp(2, a, 1)))
    F_val = (8.0 * a * B0) / ((4.0 * B0 * kv(1, a)) - (a**2 * kvp(2, a, 1)))

    # Project constants onto function space so that we can pick values later
    C_field = Function(Q)
    C_field = project(C_val, Q)
    D_field = Function(Q)
    D_field = project(D_val, Q)
    F_field = Function(Q)
    F_field = project(F_val, Q)

    # Compute analytical solution for velocity
    print("    Velocity ...")
    class VSolution(Expression):
        """Compute analytical solution for velocity"""

        def eval(self, value, x):
            x0 = x[0] * x_scaling
            x1 = x[1] * x_scaling

            r = sqrt(x0*x0 + x1*x1)

            # Simple shear:
            Ex   = [0.5*gamma*x1, 0.5*gamma*x0]
            xExx = [gamma*x0*x1*x0, gamma*x0*x1*x1]
            wx   = [gamma*x1, 0.0]

            # Constant fields evaluated at point
            C = C_field(x0,x1)
            D = D_field(x0,x1)
            F = F_field(x0,x1)

            # Calculation of velocity components
            value[0] = (( (-4.0 * D / r**4) + (2.0 * F * kv(2, r) / r**2) ) * Ex[0] + \
                       ( (-2.0 * C / r**4) + (8.0 * D / r**6) - (F * kv(3, r) / r**3) ) * xExx[0] \
                       + wx[0]) / v_scaling
            value[1] = (( (-4.0 * D / r**4) + (2.0 * F * kv(2, r) / r**2) ) * Ex[1] + \
                       ( (-2.0 * C / r**4) + (8.0 * D / r**6) - (F * kv(3, r) / r**3) ) * xExx[1] \
                       + wx[1]) / v_scaling

        def value_shape(self):
            return (2,)

    velocity_solution = VSolution()
    analytical_u      = Function(V)
    analytical_u.interpolate(velocity_solution)

    # Compute analytical solution for pressure
    print("    Pressure ...")
    class PSolution(Expression):
        """Compute analytical solution for pressure"""

        def eval(self, value, x):
            x0 = x[0] * x_scaling
            x1 = x[1] * x_scaling

            r = sqrt(x0*x0 + x1*x1)

            # Simple shear:
            xEx = gamma*x0*x1

            # Constant fields evaluated at point
            C = C_field(x0,x1)
            F = F_field(x0,x1)

            # Calculation of pressure
            value[0] = ( (-4.0 * B0 * C / r**4) + (F * kv(2, r) / r**2) ) * xEx / P_scaling

    pressure_solution = PSolution()
    analytical_p      = Function(Q)
    analytical_p.interpolate(pressure_solution)

    # Analytical compaction rate
    print("    Compaction rate ...")
    class Compaction(Expression):
        """Compute analytical solution for compaction rate"""

        def eval(self, value, x):
            x0 = x[0] * x_scaling
            x1 = x[1] * x_scaling

            r = sqrt(x0*x0 + x1*x1)

            # Simple shear
            xEx = gamma*x0*x1

            # Constant fields evaluated at point
            F = F_field(x0,x1)

            value[0] = (F * kv(2, r) / r**2) * xEx

    compaction_solution = Compaction()
    analytical_comp     = Function(Q)
    analytical_comp.interpolate(compaction_solution)

    # Store analytical solution
    output_dir  = "output/"
    extension   = "xdmf"   # "xdmf" or "pvd"
    aufile_out  = File(output_dir + "analytical_u." + extension)
    aufile_out  << analytical_u
    apfile_out  = File(output_dir + "analytical_p." + extension)
    apfile_out  << analytical_p
    acrfile_out = File(output_dir + "analytical_compaction." + extension)
    acrfile_out << analytical_comp

    # Numerical compaction rate from analytical solution
    adiv          = Function(Q)
    adiv          = project(div(analytical_u), Q)
    adiv_file_out = File(output_dir + "analytical_div." + extension)
    adiv_file_out << adiv

    # Numerical compaction rate from numerical solution
    div_u = Function(Q)
    div_u = project(div(u), Q)

    # Compute error measures
    print("**** Computing error measures ...")

    # Pressure error
    mean_p      = assemble(p*dx)/(aspect*height*height - math.pi*radius**2)
    print("     Mean numerical pressure is %g" % (mean_p))
    shifted_p   = project(p - mean_p, Q)
    analyt_norm = norm(analytical_p)
    num_norm    = norm(shifted_p)
    error_p     = (errornorm(analytical_p, shifted_p)) / analyt_norm
    print("     Pressure L2 norms: analytical %g, numerical %g, error %g" \
         % (analyt_norm, num_norm, error_p))
    if MPI.rank(comm) == 0:
        logfile.write("Pressure L2 norms: analytical %g, numerical %g, error %g\n" \
                      % (analyt_norm, num_norm, error_p))

    # Velocity error
    analyt_norm = norm(analytical_u)
    num_norm    = norm(u)
    error_u     = (errornorm(analytical_u, u)) / analyt_norm
    print("     Velocity L2 norms: analytical %g, numerical %g, error %g" \
         % (analyt_norm, num_norm, error_u))
    if MPI.rank(comm) == 0:
        logfile.write("Velocity L2 norms: analytical %g, numerical %g, error %g\n" \
                      % (analyt_norm, num_norm, error_u))

    # Compaction rate error
    analyt_norm = norm(analytical_comp)
    num_norm    = norm(div_u)
    error_c     = (errornorm(analytical_comp, div_u)) / analyt_norm
    print("     Compaction rate L2 norms: analytical %g, numerical %g, error %g" \
         % (analyt_norm, num_norm, error_c))
    if MPI.rank(comm) == 0:
        logfile.write("Compaction rate L2 norms: analytical %g, numerical %g, error %g\n" \
                      % (analyt_norm, num_norm, error_c))
    
# ======================================================================

def cylinder_integrals(field, name, param, timestep):
    """Computation of scaled integrals around the cylinder"""

    print("**** Computing %s integrals ..." % (name))

    # Output files
    radius_integral_file = "./output/radius_integral_%s_%s.txt" % (name, timestep)
    rfile = open(radius_integral_file,"w")
    sin_integral_file = "./output/sin_integral_%s_%s.txt" % (name, timestep)
    sfile = open(sin_integral_file,"w")
    cos_integral_file = "./output/cos_integral_%s_%s.txt" % (name, timestep)
    cfile = open(cos_integral_file,"w")

    # Get parameters
    radius = param['radius']

    # Integration parameters
    ntheta    = 100        # Number of angles to sample
    nradius   = 100        # Number of radial points to sample
    maxradius = 2.0*radius # Max radius to use in integration

    dtheta    = 2.0*pi / float(ntheta)
    dradius   = (maxradius - radius) / float(nradius)

    # Integral over radius, for different angle theta
    # Loop over theta
    for i in range(ntheta):
        theta = float(i) * dtheta
    
        integral = 0
        # Get points for radius range
        # Note: range has nradius+1 here in order to use last interval
        for j in range(nradius + 1):    
            r  = radius + (float(j) * dradius)
            
            x0 = r*sin(theta)
            x1 = r*cos(theta) 

            field_local = field(x0, x1)

            # Integrate field using trapezoidal rule
            if j == 0 or j == (nradius):
                integral += dradius * 0.5 * field_local
            else:
                integral += dradius * field_local

        # Scale integral by radius
        scaled_integral = integral / (maxradius - radius)
       
        # Write integrated value to file
        rfile.write("%g %g\n" % (theta, scaled_integral) )

    rfile.close()

    # Quadrupole integral (sin 2theta and cos 2theta), for different radii
    # Loop over radius
    for j in range(nradius + 1):
        r  = radius + (float(j) * dradius)                                                                   
    
        sin_integral = 0
        cos_integral = 0
        # Get points for angle range
        # Note: range has no ntheta+1 here because of periodicity
        for i in range(ntheta):                                                                             
            theta = float(i) * dtheta 
            x0 = r*sin(theta) 
            x1 = r*cos(theta)
            field_local = field(x0, x1)

            # Integrate field*r using trapezoidal rule 
            if j == 0 or j == (ntheta - 1): 
                sin_integral += dtheta * 0.5 * field_local * sin(2.0*theta)
                cos_integral += dtheta * 0.5 * field_local * cos(2.0*theta)
            else: 
                sin_integral += dtheta * field_local * sin(2.0*theta)
                cos_integral += dtheta * field_local * cos(2.0*theta)

        # Scale integral by pi
        scaled_sin_integral = sin_integral / pi                                                                 
        scaled_cos_integral = cos_integral / pi                                                                 
        # Write integrated value to file
        sfile.write("%g %g\n" % (r, scaled_sin_integral) )
        cfile.write("%g %g\n" % (r, scaled_cos_integral) )
    
    sfile.close()
    cfile.close()

# ======================================================================

def cylinder_integrals_slice(field, name, param, timestep):
    """Computation of scaled integrals around a sphere, on a 2-D slice"""

    print("**** Computing %s integrals ..." % (name))

    # Output files
    radius_integral_file = "./output/radius_integral_%s_%s.txt" % (name, timestep)
    rfile = open(radius_integral_file,"w")
    sin_integral_file = "./output/sin_integral_%s_%s.txt" % (name, timestep)
    sfile = open(sin_integral_file,"w")
    cos_integral_file = "./output/cos_integral_%s_%s.txt" % (name, timestep)
    cfile = open(cos_integral_file,"w")

    # Get parameters
    radius = param['radius']

    # Integration parameters
    ntheta    = 100        # Number of angles to sample
    nradius   = 100        # Number of radial points to sample
    maxradius = 2.0*radius # Max radius to use in integration
    x_centre  = 0.5        # X-coordinate of inclusion centre
    z_centre  = 0.5        # Z-coordinate of inclusion centre

    dtheta    = 2.0*pi / float(ntheta)
    dradius   = (maxradius - radius) / float(nradius)

    # Integral over radius, for different angle theta
    # Loop over theta
    for i in range(ntheta):
        theta = float(i) * dtheta

        integral = 0
        # Get points for radius range
        # Note: range has nradius+1 here in order to use last interval
        for j in range(nradius + 1):
            r  = radius + (float(j) * dradius)

            x0 = r*sin(theta)
            x1 = z_centre + r*cos(theta)

            field_local = field(x_centre, x0, x1)

            # Integrate field using trapezoidal rule
            if j == 0 or j == (nradius):
                integral += dradius * 0.5 * field_local
            else:
                integral += dradius * field_local

        # Scale integral by radius
        scaled_integral = integral / (maxradius - radius)

        # Write integrated value to file
        rfile.write("%g %g\n" % (theta, scaled_integral) )

    rfile.close()

    # Quadrupole integral (sin 2theta and cos 2theta), for different radii
    # Loop over radius
    for j in range(nradius + 1):
        r  = radius + (float(j) * dradius)

        sin_integral = 0
        cos_integral = 0
        # Get points for angle range
        # Note: range has no ntheta+1 here because of periodicity
        for i in range(ntheta):
            theta = float(i) * dtheta

            x0 = r*sin(theta)
            x1 = z_centre + r*cos(theta)

            field_local = field(x_centre, x0, x1)

            # Integrate field*r using trapezoidal rule
            if j == 0 or j == (ntheta - 1):
                sin_integral += dtheta * 0.5 * field_local * sin(2.0*theta)
                cos_integral += dtheta * 0.5 * field_local * cos(2.0*theta)
            else:
                sin_integral += dtheta * field_local * sin(2.0*theta)
                cos_integral += dtheta * field_local * cos(2.0*theta)

        # Scale integral by pi
        scaled_sin_integral = sin_integral / pi
        scaled_cos_integral = cos_integral / pi

        # Write integrated value to file
        sfile.write("%g %g\n" % (r, scaled_sin_integral) )
        cfile.write("%g %g\n" % (r, scaled_cos_integral) )

    sfile.close()
    cfile.close()

# ======================================================================

def plane_wave_analysis(Q, u, t, param, logfile):
    """Comparison of analytical and numerical growth rate of planar shear bands"""

    print("**** Computing diagnostics ...")

    phiB      = param['phiB']
    rzeta     = param['rzeta']
    alpha     = param['alpha']
    amplitude = param['amplitude']
    angle_0   = math.pi * param['angle_0']
    k_0       = math.pi * param['k_0']
    B0        = 1.0 / (rzeta + (4.0/3.0))

    # Compute current wavenumber and its components
    k_x0      = k_0 * sin(angle_0)
    k_y0      = k_0 * cos(angle_0)
    k_t       = Constant(sqrt(k_x0**2 + (k_y0 - k_x0*t)**2))
    angle     = Constant(math.atan2(sin(angle_0), (cos(angle_0) - t * sin(angle_0))) )
    angle_deg = angle * 180.0/math.pi
    k_x       = Constant(k_t * sin(float(angle)))
    k_y       = Constant(k_t * cos(float(angle)))

    # Compute analytical shear band growth rate
    # (1) Spiegelman 2003 pg5 (27)
    ds_dt_S03 = 2.0 * alpha * B0 * (1.0 - phiB) * (k_x * k_y) / (k_t**2 + 1.0)
    # (2) Takei & Katz (submitted) pg18 (5.26)
    ds_dt_T12 = alpha * (1.0 - phiB) * sin(2.0*angle)/(rzeta + 4.0/3.0)

    # Compute numerical shear band growth rate
    sdot      = Function(Q)
    #Katz & Takei (submitted) pg6 section 3.1:
    num_ds_dt = ((1.0 - phiB)/phiB) * div(u) / amplitude
    sdot      = project(num_ds_dt, Q)
    ds_dt     = sdot(0.0, 0.0)

    # Norm of growth rate field according to Spiegelman 2003
    dsdt_S03   = project(ds_dt_S03, V = Q)
    sinner_S03 = dsdt_S03 * dsdt_S03 * dx
    snorm_S03  = assemble(sinner_S03)

    # Norm of growth rate field according to Takei & Katz, submitted
    dsdt_T12   = project(ds_dt_T12, V = Q)
    sinner_T12 = dsdt_T12 * dsdt_T12 * dx
    snorm_T12  = assemble(sinner_T12)

    # Compute error in shear band growth rate
    ds_dt_error_S03 = (ds_dt_S03 - ds_dt) / snorm_S03
    ds_dt_error_T12 = (ds_dt_T12 - ds_dt) / snorm_T12

    # Compute analytical shear band growth
    s_t = alpha * B0 * (1.0 - phiB) * numpy.log((1.0 + k_0**2) / \
          (1.0 + float(k_t)**2))

    # Compute analytical amplitude
    A_t = exp(s_t)

    print("t = %f: angle(t) = %f; k(t) = %f; s(t) = %f; A(t) = %f" \
         % (t, angle_deg, k_t, s_t, A_t))
    if (MPI.rank(comm) == 0):
        logfile.write("t = %f: angle(t) = %f; k(t) = %f; s(t) = %f; A(t) = %f\n" \
                      % (t, angle_deg, k_t, s_t, A_t))
    print("ds_dt(t) (Spiegelman 2003)   = %f; num_ds_dt(t) = %f; rel_error = %f" \
         % (ds_dt_S03, ds_dt, ds_dt_error_S03))
    if (MPI.rank(comm) == 0):
        logfile.write("ds_dt(t) (Spiegelman 2003)   = %f; num_ds_dt(t) = %f; rel_error = %f\n" \
                      % (ds_dt_S03, ds_dt, ds_dt_error_S03))
    print("ds_dt(t) (Takei & Katz subm.) = %f; num_ds_dt(t) = %f; rel_error = %f" \
         % (ds_dt_T12, ds_dt, ds_dt_error_T12))
    if (MPI.rank(comm) == 0):
        logfile.write("ds_dt(t) (Takei & Katz subm.) = %f; num_ds_dt(t) = %f; rel_error = %f\n" \
                      % (ds_dt_T12, ds_dt, ds_dt_error_T12))

# ======================================================================

# EOF
