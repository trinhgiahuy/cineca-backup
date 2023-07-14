#include<stdlib.h>
#include<stdio.h>

#define N 10000000

void main() {

   int i;
   float *a, *b, *c;
   a = (float*)malloc(N*sizeof(float));
   b = (float*)malloc(N*sizeof(float));
   c = (float*)malloc(N*sizeof(float));

   for (i=0;i<N;i++) {
      b[i]=1.0f;
      c[i]=2.0f;
   }

   for(i=0;i<N;i++)
      a[i] = b[i] + c[i];

}

   
  
