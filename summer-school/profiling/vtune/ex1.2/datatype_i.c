#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

int main()
{

double time,time1;
int t,i,j,k,totalsize,size = 500,repetition=40;


totalsize = size * size * size;

double * a =         (double *) _mm_malloc(totalsize * sizeof(double),64);
signed char * b =    (signed char *) _mm_malloc(totalsize * sizeof(signed char),64);
double * c =         (double *) _mm_malloc(totalsize * sizeof(double),64);

for(int i=0;i<totalsize;i++)
{
	a[i] = i+1;
	b[i] = i%2;
}

time = omp_get_wtime();
for(int t=0;t<repetition;t++)
	for(int i=0;i<totalsize;i++)
	{
			c[i] = a[i] * b[i];
	}

time = omp_get_wtime() - time;

printf("TIME int8 %e \n",time/repetition);

_mm_free(a);
_mm_free(b);
_mm_free(c);

}