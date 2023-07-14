EX1.3 : Front End Bound
====================================

Use Makefile to compile and build feb.f90:

Before compilation load the follow module:

    module load profile/global
    module load intel/pe-xe-2018--binary vtune/2018

and after run 

    make all

Launch the exe.

Vtune
-----
Use Intel VTune profile to obtain a general characterization for exe.

To launch vtune:

    amplxe-gui

Select ***General Analisys** and follow the instruction on the screen.