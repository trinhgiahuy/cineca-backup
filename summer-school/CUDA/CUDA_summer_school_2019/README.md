# Instructions for CUDA course

## Example 1. Simple vector addition
1. Take a look at the C program for adding vectors and try to see how it works.
Then compile and run the program
```bash
gcc -o add vecadd.c
./add
```

2. We will write a CUDA version of the program by doing the following:
 - make the add function a CUDA kernel, removing the loop;
 - insert the lines to allocate memory on the device;
 - insert lines to copy the input arrays to the device;
 - call the GPU kernel;
 - copy the output data from the device;
 - free both host and device allocated memory;
 - rename the file from vecadd.c to vecadd.cu

To avoid tedious typing the memory allocation code in CUDA has simple been commented in the C code.

3. Compile with th CUDA compiler and run the program:
```bash
module load autoload cuda
nvcc -arch=sm_37 -o addcuda vecadd.cu
./addcuda
```
Don't forget the *-arch* flag of nvcc! 
Does it gave the same result as the non-accelerated version?

4. Profile the CUDA program with nvprof
```
nvprof ./addcuda
```
What are the most expensive parts of the program?

## Example 2. Jacobi integration

### Description
With the example above it is difficult test the advantages of using GPUs. In this example we will instead 
illustrate how using accelerators can dramatically improve the performance of a common algorithm in HPC.

1. The jacobi algorithm is a common method for solving partial differential equations. In the CUDA directory you will find a file called `jacobi.c` which contains a C program to perform the jacobi algorithm on a 2D grid of numbers.
This program also implements boundary conditions to mimic heat flow from the bottom of the grid to the top.
Compile and run the program with various grid sizes to understand how it works, perhaps also using the time command to see how long it takes:
```bash
gcc -o jacobi jacobi.c -lm
time ./jacobi 200 200
```

2. You should now look at the jacobi program and identify the parts that should be run on the GPU.
In particular there are two double for loops:
. the loop needed to update the grid for the next iteration
. a reduction in order to find the "norm" value in order to test for convergence.

For the moment we make a kernel of the update loop since reduction in CUDA is not trivial (unlike MPI or OpenMP).

3. Rewrite the program with the update as a CUDA kernel as follows:
- Allocate memory in the GPU for the grid and updated grid using cudaMalloc (just uncomment the lines in the C program)
- Create a CUDA kernel for the update loop. This can be done by copying the host code and using the CUDA thread mechanism as used in the previous example.
- copy the grid data into the device, call the kernel, copy the results back. You will want something like this in the main loop:

```bash
    cudaMemcpy(d_grid,d_grid_new,...,cudaMemcpyDeviceToDevice);
    stencil_sum<<<numBlocks,blockSize>>>(d_grid,d_grid_new,nx,ny);
    cudaMemcpy(grid,d_grid_new,...,cudaMemcpyDeviceToHost);
```
Note that the first cudaMemcpy copies the updated data into the grid to update *within* the device, avoiding a copy back to the host.

 



