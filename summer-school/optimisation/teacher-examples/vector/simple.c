#include <stdlib.h>

#define N 1000000

void main()
{

   int i;
   float *a,*b,*c;

   a=(float*)malloc(sizeof(float)*N);
   b=(float*)malloc(sizeof(float)*N);
   c=(float*)malloc(sizeof(float)*N);

   // assume we use this fucntion to initialise a,b
   init(a,b);
   printf("Initialised..\n");

   for (i=0;i<N;i++) 
      a[i]=b[i]*c[i];


}
