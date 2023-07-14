#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=gll_usr_prod
#SBATCH --job-name=multiple_queues
#SBATCH --error=err-multiple_queues
#SBATCH --output=out-multiple_queues
#SBATCH --account=train_scB2019

cd $SLURM_SUBMIT_DIR

module purge
module load profile/archive
module load python/2.7.12
module load intel/pe-xe-2017--binary
module load intelmpi/2017--binary
module load mpi4py/1.3.1--intelmpi--2017--binary

rm -f inputs/*.out

srun ./multiple_queues.py serial_program.x inputs
