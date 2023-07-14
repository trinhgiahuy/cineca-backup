#!/bin/bash

#-------------------- hw section -------------------------------

#### Ask for a number of nodes each with cpus "ntasks-per-node" 
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --mem=
 
#### Maximum length of the job (hh:mm:ss)
#SBATCH --time=

#-------------------- partition system -------------------------------

#### Partition of submission
#SBATCH --partition=


# --------------- accounting/budget  -----------------------------

#### account name (type saldo -b)
#SBATCH --account= 

# ---------------- other info ------------------------------------

#### Files for standard output and error
#SBATCH --output=
#SBATCH --error=

#### Job name
#SBATCH --job-name=hello_my_friend

#### send email to the following address
#SBATCH --mail-user= 

#### send email after abort or end
#SBATCH --mail-type= 

# -----------end of SLURM keywords section -------------------------

hostname 

echo 'hello world' 

sleep 4





