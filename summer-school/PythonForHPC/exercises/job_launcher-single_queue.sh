#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=gll_usr_prod
#SBATCH --job-name=single_queue
#SBATCH --error=err-single_queue
#SBATCH --output=out-single_queue
#SBATCH --account=train_scB2019

cd $SLURM_SUBMIT_DIR

module purge
module load profile/archive
module load python/2.7.12
module load intel/pe-xe-2017--binary

rm -f inputs/*.out

./single_queue.py serial_program.x inputs
