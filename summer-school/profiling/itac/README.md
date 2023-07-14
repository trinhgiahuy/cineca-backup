# Profiling MPI with APS and ITAC

## Introduction

Intel provides some simple tools for analyzing the performance of parallel programs: APS (Application Performance Snaphsot) and ITAC (Intel Trace Analyzer and Collector). It is recommended you
start with APS to get an overall summary of the performance, then pass to ITAC to analyze in more
detail the MPI performance.

## APS
A job script is provided on how to run the tool. You can use mpirun or srun in this case.

## ITAC

### Usage

In a job script use the following commands:

```bash
module load autoload intelmpi
source $INTEL_HOME/itac_latest/bin/itacvars.sh intel64 #Intel 18
mpirun -trace executable
```
Note that if use *srun* instead of mpirun you cannot use the above method. Instead you have
two choices
1.  re-compile and link the program with the ```-trace``` option of intel C/FORTRAN.
2. or link in the trace library during execution:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTEL_HOME/itac/latest/slib
export LD_PRELOAD=$INTEL_HOME/itac/latest/slib/libVT.so
export VT_LOGFILE_FORMAT=stfsingle
export VT_PCTRACE=5
srun executable
```

This will create various files including one called <executable>.stf.
To run the profiler, use the traceanalyzer command:
```bash
module load autoload intelmpi
traceanalyzer executable.stf
```
This will open up a graphics window with MPI tracing results.

## Profiling DL_POLY
You can profile your own program of course but to demonstrate ITAC we have set up a demonstration based on the 
DLPOLY molecular dynamics program. This application was chosen because it exists in two versions, 1.9 and 4.x, which rely on two different parallelization schemes: the former uses a replicated data strategy while the second (more efficient) on domain decomposition.

### Tasks
Modify the job script ```itac.slurm`` to run DL_POLY 1.9 with, say, 8 and 16, tasks. In particular, look at
1. The ratio of time spent in MPI and serial calls.
2. The MPI calls which have consumed the most time.
3. The load balancing of the tasks.

Now repeat with DL_POLY 4.x. How does the profile compare with that of DL_POLY 1.9?



