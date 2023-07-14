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
source $INTEL_HOME/performance_snapshots/apsvars.sh


# copy input data
cp ./data/* .

aps srun  $exe

# Analyse with
# aps --report=aps_result_<date>
# Also for MPI programs
# aps-report -D -t aps_result_20181113

