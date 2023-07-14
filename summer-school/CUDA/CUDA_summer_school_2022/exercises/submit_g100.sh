#!/bin/bash
#SBATCH --job-name first_kernel
#SBATCH -N1 --ntasks-per-node=2
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1
#SBATCH --account=<your_account_name>
#SBATCH --partition=g100_usr_interactive


# if you want to compile also
module load hpc-sdk/2021--binary
nvcc -arch=sm_70 -o first_kernel 01_cuda_first_kernel.cu  -run

# OR
# if you only want to run an executable, loading hpc-sdk is not necessary
./first_kernel
