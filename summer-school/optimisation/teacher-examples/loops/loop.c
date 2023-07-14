#include <stdio.h>
#include <stdlib.h>

#define N 10000
void main() {

   int a[N],b[N], c[N];
   int i;

   for (i=0;i<N; i++) {
      b[i]=i;
      c[i]=i;
   }

   for (i=0; i<N; i++) {
      a[i]=b[i]+c[i];
   }

}


   
