#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>

#define ITMAX 10000000
void main(int argc, char *argv[]) {
   
int i,it,n;
double *a,*b, *c;
struct timeval t1,t0;
long long elapsed;

if (argc>1) { 
  n=atoi(argv[1]);
printf("n=%d\n",n);
}
 else {
printf("No matrix size given\n");
   exit(1);
}
a=(double*)malloc(n*sizeof(double));
b=(double*)malloc(n*sizeof(double));
c=(double*)malloc(n*sizeof(double));
 
for(i=0;i<n;i++) {
   a[i]=cos(i*0.1);
   b[i]=cos(i*0.1);
   c[i]=0.0;
}

gettimeofday(&t0,NULL);
for(it=1;it<=ITMAX;it++){
  for(i=0;i<n;i++)
     c[i]=a[i]+b[i];

   b[n-1]=sin(b[n-1]);
}
gettimeofday(&t1,NULL);
elapsed = (t1.tv_sec-t0.tv_sec)*1000000LL + t1.tv_usec-t0.tv_usec;

printf("%lf\n",(double)elapsed/1000000);

free(a);
free(b);
free(c);

}
