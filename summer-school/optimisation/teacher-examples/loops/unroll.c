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

  for(i=0;i<N;i+=4) {
     a[i] = b[i] + c[i];
     a[i+1] = b[i+1] + c[i+1];
     a[i+2] = b[i+2] + c[i+2];
     a[i+3] = b[i+3] + c[i+3];
  }
}

   
  
