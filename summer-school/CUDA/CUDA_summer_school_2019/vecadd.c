#include <stdlib.h>
#include <stdio.h>

#define N (2048*2048)

void add( int *a, int *b, int *c) {

   int i;
   for(i=0;i<N;i++)
      c[i]=a[i]+b[i];

}


int main( void ) {
 int *a, *b, *c; // host copies of a, b, c
 int size = N * sizeof( int ); // we need space for N integers
 int i;
 long sum;

// Allocate memory for the three arrays
 a = (int*)malloc( size );
 b = (int*)malloc( size );
 c = (int*)malloc( size );

//CUDA only
// allocate device copies of a, b, c
// int *dev_a, *dev_b, *dev_c; // device copies of a, b, c

//  cudaMalloc( (void**)&dev_a, size );
//  cudaMalloc( (void**)&dev_b, size );
//  cudaMalloc( (void**)&dev_c, size );
//----

//Initialise a,b arrays
for (i=0;i<N;i++) 
   a[i]=b[i]=1;


// Call add function
add(a,b,c);

// Do a reduction on C
sum=0;
for (i=0;i<N;i++) 
  sum += c[i];

printf("sum=%ld\n",sum);

 free(a); free(b); free(c);
 return 0;
}


