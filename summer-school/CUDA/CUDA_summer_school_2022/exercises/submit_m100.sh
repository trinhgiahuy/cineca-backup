#!/bin/bash
#SBATCH --job-name first_kernel
#SBATCH -N1 --ntasks-per-node=2
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1
#SBATCH --account=<your_account_name>
#SBATCH --partition=m100_usr_prod
#SBATCH --qos=m100_qos_dbg            

module load hpc-sdk

nvcc -arch=sm_70 -o first_kernel 01_cuda_first_kernel.cu  -run
