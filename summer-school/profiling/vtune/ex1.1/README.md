EX1.1 : Matrix Matrix Multiplication
====================================

Use Makefile to compile and build four MMM exe:

- matrixAVX.x : vectorized MMM handwritten
- matrixNoVec.x:  no-vectorized MMM handwritten  
- matrixAVXdGEMM.x: BLAS MMM in double precision
- matrixAVXsGEMM.x: BLAS MMM in single precision

Before compilation load the follow module:

    module load profile/global
    module load intel/pe-xe-2018--binary mkl/2018--binary vtune/2018

and after run 

    make all

Check the performances.

Report file
---
Use *launch.sh* to obtain the **time to solution** for each exe and after analyze the difference among *.optrpt file.

Vtune
-----
Use Intel VTune profile to obtain detailed performance characterization for each exe.

To launch vtune:

    amplxe-gui

Select **HPC Performance Charaterization** and follow the instruction on the screen.