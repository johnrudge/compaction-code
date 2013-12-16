README for compaction-code

Aim:
====
This code computes advection and compaction in a porous medium under
simple shear. There is an option to run with a regular rectangular mesh, 
or with a mesh that has a circular torque-free inclusion.

Running:
========
run using:
> python main.py param.cfg

Contents:
=========
* main.py             : main code
* analysis.py         : library of diagnostics and benchmark quantities
* core.py             : library of core functions to deal with input and output of data, norms, etc
* mesh_gen.py         : code that interfaces with gmsh to create a rectangular mesh with a cylinder
* physics.py          : library of viscosity formulations and initial porosity fields
* param.cfg           : list of run parameters, can be copied to local run directory
* param_uniform_cylinder.cfg : list of run parameters for cylinder models with uniform initial porosity
* param_planewave.cfg : list of run parameters for plane wave benchmark, no cylinder
* param_perlin.cfg    : list of run parameters for initial noise case, no cylinder
* param_random_cylinder.cfg  : list of run parameters for cylinder models with random initial porosity, read in from file


Creating meshes and initial porosity fields:
============================================
* make_mesh.py         : create mesh, parameters provided in param_make_mesh.cfg
* make_initial_porosity.py : create coarse initial porosity field on large mesh, parameters provided in 
  param_make_initial_porosity.cfg
* interpolate_initial_porosity.py : interpolate coarse initial porosity field onto mesh used in numerical
  simulation, parameters provided in param_interpolate_initial_porosity.cfg
* param_make_mesh.cfg  : used by make_mesh.py
* param_make_initial_porosity.cfg : used by make_initial_porosity.py
* param_interpolate_initial_porosity.cfg : used by interpolate_initial_porosity.py
* run_preprocessing.py : example of pre-processing workflow; adjust file paths as necessary.


Note:
=====
This code was originally in version control in the fenics_magma repository, but has now been moved to its own repository.
