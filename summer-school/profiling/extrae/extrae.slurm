#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks=8
#SBATCH --time=30:00
#SBATCH --partition=gll_usr_prod
#SBATCH --account=cin_staff
##SBATCH --account=train_scB2018


exe=./jacobi

#  set EXTRAE environment
module load autoload extrae
source ${EXTRAE_HOME}/etc/extrae.sh
cp ${EXTRAE_HOME}/share/example/MPI/extrae.xml .
export EXTRAE_CONFIG_FILE=extrae.xml

module load autoload intelmpi
mpirun -genv LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so -n 8 $exe 100 100

# to analyse do the following in an interactive shell (preferably rcm)
# module load extrae 
# mpi2prv -f TRACE.mpits -e ./jacobi  -o trace.prv
# then run paraver
# module load paraver
# wxparaver jacobi.prv
