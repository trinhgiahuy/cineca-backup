## Setup
Installation steps on GALILEO CINECA cluster

**clone the project**

    git clone https://gitlab.hpc.cineca.it/training/summer-school/tree/master/hpc-numerical-libraries-course/petsc

**load the environment**

    module load autoload petsc

**move to an example directory**
   
In the guided_example folder you find an empty main with the text of the exercise. The solution is in the solution_example folder.

    cd https://gitlab.hpc.cineca.it/training/summer-school/tree/master/hpc-numerical-libraries-course/petsc/X_petsc_xxx/guided_example
    
**compile the example**

    make

**run the example**

    mpirun -np <n> <executable> <flags>
