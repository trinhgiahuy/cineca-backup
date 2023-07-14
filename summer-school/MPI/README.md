# Training MPI

This tutorial will guide you through the solution of our MPI exercises. Each exercise is described in the text section and solved in the solution section. We recommend you to approach the exercises in three different steps:

**Step 1**: study carefully the text of the proposed problem, write your own solution, and then compare it with the solution we propose in the solution section. If that's too difficult:

**Step 2**: look at the proposed skeleton (if available) and try to complete the blanks. Read carefully all the comments and try to answer mentally to all the questions proposed. If that's too difficult:

**Step 3**: read carefully the proposed solution, try to understand its content as much as possible, then compile it, run it and try to understand how it works and why. 

Solutions of the exercises are provided in both C and Fortran languages. In the text section you can find hints to help  you solve the exercise and the MPI functions/routines API needed to solve the exercise.

Through the course of this tutorial, you will find questions on both the text and solution sections, we encourage you to answer them to improve your understanding of the problem.

The exercises are organized as following:

**1**: first exercise: say Hello to the MPI World ;-)

**2-6**: exercises on point-to-point communications

**7-9**: exercises on collective communications

**10-12**: exercises on communicators and datatypes

# How to compile and run the exercises on Galileo100

- After login, load the compiler module (only once for the whole session):
```
module load autoload intelmpi
```
- copy the jobtemplate.slurm file to the folder with your solution to the exercise and go there:
```
cp jobtemplate.slurm /path/to/solution
cd /path/to/solution
```
- Compile depending on your programming language. Name your executable with the flag "-o"

For C:
```
mpiicc solution_xx.c -o myexe
```
For Fortran:
```
mpiifort solution_xx.f90 -o myexe
```
- Edit the jobscript jobtemplate.slurm where indicated, with the name of the executable to launch. If required by the exercise, substitute "srun -n 4" with "srun -n 2". After that, submit the job with the command:
```
sbatch jobtemplate.slurm
```

- When the job is over, check the .err file for any eventual error during the execution, and the file .out for the standard output, to be compared with your expectations.


# Rules and tips for approaching the exercises

1. Don't panic!! Thinking in MPI is hard if you are not used. Don't be afraid of failing one exercise, look at the solution if necessary and move your way backwards from there.
2. Think in parallel! Remember that your code is executed from the first to the last line indipendently by all the tasks involved, and it's up to you to make them communicate. If you can adapt your way of thinking, you are more than halfway done.
3. You can skip one or more exercises to try different topics, and return to them later. If an exercise is obsessing you, try something else and return there in another moment. You decide the pace and the order!
4. If you plan to start one exercise from scratch, write down your workplan first. Create and complete the idea you have in mind before starting to convert it into code. A good pseudo-algorithm could be enough to show that you got padronance of the parallel thinking concept.
5. Exercises are not necessarily meant to be finished before the end of the school. Take them at home, continue to work on them whenever you get the chance, until you are satisfied with the outcome.
6. The teachers are here for you, don't be afraid to ask them any question you have while they are here with you!
7. The slides are also here for you. You are not supposed to remember by memory every single MPI function/routine you may need. Keep the slides close to you and consult them whenever you need it. Also the MPI forum link and even Google are your friends!
