#include <stdio.h>

#define N 10000000


void main() {

   int i,j,K=20;
   int a[N][N],b[N][N],c[N][N], d[N][N];

for (i = 0; i < N; i = i + 1)
   for (j = 0; j < N; j = j + 1)
      a[i][j] = 2 * b[i][j];

for (i = 0; i < N; i = i + 1)
    for (j = 0; j < N; j = j + 1)
	c[i][j] = K*b[i][j]+ d[i][j]/2;
}



