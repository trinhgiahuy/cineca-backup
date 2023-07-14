# EXTRAE trace and profile tool

# Instructions

1. Do a git pull then go into the extrae directory
```bash
git pull
cd  profiling/extrae
```

2. Run any parallel program using the extrae.slurm jobscript as an example.
Note that the run should be fairly short or otherwise the resulting trace file will be large.
Alternatively, if in an RCM interactive session just run the jobscript (if < 36 cores)
```
./extrae.slurm
``
This will also generate a number of files to be used by Extrae.

3. Analyse the trace file with paraver as follows:
```bash
 module load extrae 
mpi2prv -f TRACE.mpits -e ./jacobi  -o trace.prv
 module load paraver
 wxparaver jacobi.prv
```
4. As examples we have provided 3 MPI files with different MPI calling patterns - compare and contrast the traces/
