EX1.2 : Bandwith
====================================

Use Makefile to compile and build AXPY with three different datatype for one array involved in multiplication:

- dbl.xx : B is double datatype
- int.xx : B is integer datatype
- flt.xx : B is float datatype

Before compilation load the follow module:

    module load profile/global
    module load intel/pe-xe-2018--binary vtune/2018

and after run 

    make all

Check the performances.

Vtune
-----
Use Intel VTune profile to obtain detailed bandwidth characterization for each exe.

To launch vtune:

    amplxe-gui

Select **Memory access Analysis** and follow the instruction on the screen.