#include <stdlib.h>
#include <stdio.h>


// CUDA kernel
// will run on device

#define N (2048*2048)


 __global__ void add( int *a, int *b, int *c) {

 int index = threadIdx.x + blockIdx.x * blockDim.x;

 if (index <N)  
   c[index] = a[index] + b[index];

}


#define THREADS_PER_BLOCK 512

int main( void ) {
 int *a, *b, *c; // host copies of a, b, c
 int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
 int size = N * sizeof( int ); // we need space for N integers
 int i;
 long sum;

 // allocate device copies of a, b, c
 cudaMalloc( (void**)&dev_a, size );
 cudaMalloc( (void**)&dev_b, size );
 cudaMalloc( (void**)&dev_c, size );

 // allocate host arrays
 a = (int*)malloc( size );
 b = (int*)malloc( size );
 c = (int*)malloc( size );

//Initialise a,b arrays
for (i=0;i<N;i++)
   a[i]=b[i]=1;


 // copy inputs to device
 cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice);
 cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice);

 // launch add() kernel on GPU, passing parameters
 add<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( dev_a, dev_b, dev_c);

 // copy device result back to host copy of c
 cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

// Do a reduction on C
sum=0;
for (i=0;i<N;i++)
  sum += c[i];

printf("sum=%ld\n",sum);


 // Free all data
 cudaFree( dev_a);cudaFree( dev_b);cudaFree( dev_c);
 free(a); free(b); free(c);
 return 0;
}


