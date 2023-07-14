# debugging tutorial

## Instructions
In this tutorial we will debug a program, a poisson solver, which has two types of errors:
1. An error in the serial code giving rise to a segmentation fault
2. An error in the MPI which gives rise to a deadlock

The first error can be detected with gdb but you will need totalview for the second

## Compiling and running the program
```bash
cd debugging/poisson
module load autoload openmpi
make
```
In order to run the program, ask for an interactive session and use mpirun
```bash
srun -N1 -n4 -t 1:00:00 -A train_scB2019  -p gll_usr_prod --pty bash
ulimit -c unlimited
mpirun -np 4 ./poisson.exe
```
The program should crash with a segmentation fault.

## Finding the first error
The first error can be found easily with gdb and one of the core files.
Once you have located the line responsible you can delete it (it is not needed), compile and run the program again.

## Debugging the second error with totalview

The second error is best found with totalview.
Here are the instructions:
Using the interactive session as before do the following:

```bash
srun -N1 -n4 -A train_scB2019 -p gll_usr_prod -t 1:00:00 --pty bash
module load autoload openmpi
module load totalview
totalview&
```
Use OpenMPI as the MPI system and run with 4 MPI ranks.
When you have found the error, delete the line, compile and re-run.

## Using PMPI to determine the number of MPI calls in the program
Try using PMPI to determine how often MPI_Sendrecv is called in the program.
To help we supply a simple example in the PMPI directory.
NB: If using srun (instead of mpirun) be careful to export ALL variables in the command line, e.g.
```bash
srun -n2 --export=LD_PRELOAD=./libmympi.so,ALL ./myprog
```

