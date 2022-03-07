#!/usr/bin/env python3

# ======================================================================
# runfile.py
#
# Contains different settings when using iterative solver for Stokes
#
# Run using:
#     python runfile.py
#
# WARNING: THE ITERATIVE SOLVER+PRECONDITIONER IS NOT EFFICIENT YET
#          WE'RE WORKING ON FIGURING THIS OUT
#
# Sander Rhebergen, University of Oxford
# Laura Alisic, University of Cambridge
# Garth Wells, University of Cambridge
#
# Last modified: 21 March 2013 by Sander Rhebergen
# ======================================================================


import os, sys

#===========================================================================================

run1 = 'python3 main.py param.cfg \
--petsc.ksp_monitor_true_residual \
--petsc.ksp_type bcgs \
--petsc.pc_type fieldsplit \
--petsc.pc_fieldsplit_detect_saddle_point \
--petsc.pc_fieldsplit_type schur \
--petsc.pc_fieldsplit_schur_fact_type upper \
--petsc.pc_fieldsplit_schur_precondition user \
--petsc.fieldsplit_0_ksp_type preonly \
--petsc.fieldsplit_0_pc_type hypre \
--petsc.fieldsplit_0_pc_hype_type boomeramg \
--petsc.fieldsplit_0_pc_hype_boomeramg_max_iter 1 \
--petsc.fieldsplit_0_pc_hype_boomeramg_rtol 0.0 \
--petsc.fieldsplit_1_ksp_type preonly \
--petsc.fieldsplit_1_pc_type jacobi '

run2 = 'python3 main.py param.cfg \
--petsc.ksp_monitor_true_residual \
--petsc.ksp_type bcgs \
--petsc.pc_type fieldsplit \
--petsc.pc_fieldsplit_detect_saddle_point \
--petsc.pc_fieldsplit_type schur \
--petsc.pc_fieldsplit_schur_fact_type upper \
--petsc.pc_fieldsplit_schur_precondition user \
--petsc.fieldsplit_0_ksp_type preonly \
--petsc.fieldsplit_0_pc_type lu \
--petsc.fieldsplit_1_ksp_type preonly \
--petsc.fieldsplit_1_pc_type lu '

#===========================================================================================

runs = [run2]

for rr in runs:
    os.system(rr)
