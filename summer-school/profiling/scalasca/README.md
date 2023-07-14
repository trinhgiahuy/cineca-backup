# Profiling exercises with Scalasca

*Remember to use -X with ssh when connecting to Galileo*

In this exercise we will analyze the performance of two versions of the same moecular
dynamics program, DL_POLY, but parallelized with different parallelization strategies.
The older version, DL_POLY 1.9, uses a replicated data strategy while the second version, 
DL_POLY v4.03, uses a more efficient domain decomposition scheme. Both programs have been
prepared for SCALASCA analysis.

### Copy files
The executables for DLPOLY are not available in this repository but can be copied
using this script.
```bash
./copy-dlpoly.sh
```
DLPOLY.X.sc = DL_POLY 1.9 (replicated data)
DLPOLY.Z.sc = DL_POLY 4.03 (domain decomposition) 

Both require the same input files which are also copied.

#### Procedure


step 1. Copy the executables in the profiling/scalasca directory:
```bash
cd profiling/scalasca
./copy-dlpoly.sh
```
THe DLPOLY.X.sc is the executable for the DL_POLY 1.9 version, while the other is for version
4.04. Both have been compiled for Broadwell with Scalasca v2 wrapper.

step 2. Run the Scalasca versions of DL_POLY. 
The input for both versions is found in the input/ subdirectory. You need to copy
the three files in your execution directory first.

```bash
mkdir run1
cp DLPOLY.X.sc run1
cp input/* run1
cd run1
```
Create a job script file like the following
```bash
#!/bin/bash
#SBATCH -N1 -n8
#SBATCH -t 30:00
#SBATCH -A train_scB2019
#SBATCH -p gll_usr_prod

module load autoload scalasca

scan srun -n 8 ./DLPOLY.X.sc
```
step 3. Analyze the output

At the end of the simulation you should find a directory called `scorep_DLPOLY_8_sum`
or similar. This can be analyzed with SCALASCA.
```bash
module load autoload scalasca
scalasca -examine -s <experiment name>  # without graphics
# or
square <experiment name>
# e.g.
square scorep_DLPOLY_8_sum
```


### Questions
1. Try running both versions at, for example, 4,8,16,36 or more cores. Which version is more
efficient?
2. What MPI commands do the two programs use?
3. What MPI command(s) causes the scaling problems in either versions.


