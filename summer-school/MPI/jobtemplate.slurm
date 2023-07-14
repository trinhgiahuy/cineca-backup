#!/bin/bash

#SBATCH --job-name=my_test
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --account=tra23_MpiOmp
#SBATCH --output=output_%j.out
#SBATCH --error=output_%j.err
#SBATCH --reservation=s_tra_mpiomp

PROGRAM=myexe    ### EDIT THIS with the name of your compiled executable!! Have you compiled it first?

module load autoload intelmpi
srun -n 4 ./${PROGRAM}
