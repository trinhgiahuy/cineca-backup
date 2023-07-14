#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --time=30:00
#SBATCH --partition=gll_usr_prod
#SBATCH --account=cin_staff
##SBATCH --account=train_scB2019

module load autoload intelmpi

#  DLPOLY v1.9 uses replicated data
exe=./bin/DLPOLY.X

#  DLPOLY 4.0 uses domain decomposition
#exe=./bin/DLPOLY.Z

#  set ITAC environment
source $INTEL_HOME/itac_latest/bin/itacvars.sh intel64

# copy input data
cp ./data/* .

# here use mpirun here and not srun
mpirun -trace  $exe

# to analyse do the following in an interactive shell
# source $INTEL_HOME/itac/2017.0.020/intel64/itac/itacvars.sh
# traceanalyzer DLPOLY.stf
