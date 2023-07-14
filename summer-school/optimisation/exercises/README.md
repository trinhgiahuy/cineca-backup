# Optimisation Exercises


## General instructions

Remember to use interactive mode when running all the tests:
```bash
srun -N1 -n1 -p gll_usr_dbg --time=1:00:00 -A <account_no> --pty bash
```



## Memory bandwidth using the Stream and Triad benchmarks
Directory: exercises/stream

In this exercise you can test the memory bandwidth of the system you are using with stream or triad benchmarks,
[stream benchmark](https://www.cs.virginia.edu/stream/).

The idea is that a simple loop with an array is used to measure the memory bandwidth. To ensure that the caches are not being used the array size should be *at least 4x the size of the sum of all the last-level caches* used in the run.

### Instructions

1. Problem 1 - Memory bandwidth
Look at the stream.c or stream.f program files and locate the parameter for the array size and set it so that it is big enough to test the memory bandwidth and not the cache size.
For you information,

```
Galileo Broadwell: 
L2 cache:              256K
L3 cache:              46080K

```
Compile and run the program. What is the memory bandwidth on Marconi Broadwell or Marconi KNL?
Repeat the exercise but this time use the threaded version of stream by setting the number of OMP_THREADS:
```bash
export OMP_NUM_THREADS=<n>
```
where <n> is 2,4,8,etc.

2. Problem 2 - cache sizes (advanced)
To see the effect of cache sizes on the stream run times you can use the stream or triad benchmarks but this time
by running repeatedly the program with different array sizes.
Can you estimate the L1 cache size?


## Vectorisation

Directory: exercises/vector
1. Problem 1 
In the vector/ directory you will find various Fortran and C programs named test1.f, test2.c, etc. BEFORE compiling them try to understand if they will vectorise or not. (Note that for the C programs you may have to specify the -std=c99 flag). 

2. Problem 2
You will find a file vecadd.f90 or vecadd.c which does a simple vector addition. Compile the program with and without vectorisation and in single and double precision:
```bash
ifort -no-vec -o no-vec vecadd.F90
ifort -O2 -o vec vecadd.F90
```
Or for C:
```bash
icc -O2 -no-vec -o no-vec vecadd.c -lm
icc -O2 -o vec vecadd.c -lm
```
Depending on the time you have try the vector size n with these values, n=10,100,500,1000 etc.
Discussion
Thinking of caches, can you explain these results ? 




