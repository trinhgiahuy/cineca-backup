/**
 * 
 * Matrix multiplication Benchmark
 * 
 * This software measure time to solution and FLOPS provided by 
 * a different way to calculate matrix multiplication.
 * 
 * The way actualy implemented are two:
 * - 3 nested loop
 * - cblas_dgemm lib call (if you define MKL_DGEMM env var)
 * 
 */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>



#ifdef MKL_DGEMM 
  #include "mkl.h"
  #define _MTYPE double
#elif MKL_SGEMM 
  #include "mkl.h"
  #define _MTYPE float
#else
  #define _MTYPE double
#endif


struct matrix
/*
 * Structure of matrix using in program.
 * More usefull because contains information of dimension
 * in var rows and cols and the pointer to real matrix array
 */
{
  int rows;
  int cols;
  _MTYPE *matrix;
};


double cclock()
/*
 * Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970)
 * It can be used to measure elapsed time within a code. For example:
 * 
 * tstart = cclock()
 * ....
 * tstop = cclock()
 * print("Elapsed time in seconds: %.3g", tstop - tstart);
 */
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}


void multiplication(struct matrix A,struct matrix B,struct matrix C)
/*
 * This function execute the matrix multiplication calculus.
 * There are two possible way, three nested loop and MKL cblas_dgemm
 * lib. For using library define in make env var MKL_DGEMM
 * 
 */
{
  
  
  #ifdef MKL_DGEMM
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, B.rows, 1, 
	          A.matrix, A.cols, B.matrix, B.cols, 1, C.matrix, C.cols);

  #elif MKL_SGEMM
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.cols, B.rows, 1, 
	          A.matrix, A.cols, B.matrix, B.cols, 1, C.matrix, C.cols);
  #else
  
  for(int i=0; i<C.rows;i++)
  {
    for(int j=0;j<C.cols;j++)
    {
      for(int r=0;r<A.cols;r++)
      {
	       C.matrix[j+(i*C.cols)]= C.matrix[j+(i*C.cols)] + ( A.matrix[r+(i*A.cols)] * B.matrix[j+(r*B.cols)] );
	
      }    
    }
  }
  
  #endif
  
}



int main ( int argc, char *argv[] )
/**
 * 
 * Main program, init matrix and calculate their product.
 * Variable dimension specified tthe dimension of square matrix
 * and the subroutine calculate the time spend in multilication
 * and the Flops powered.
 * 
 */
{	
  struct matrix A,B,C;
  double sec;
  int dimension= 1000;
  
  A.rows = dimension;
  A.cols = dimension;
  A.matrix = malloc((A.rows*A.cols)*sizeof(_MTYPE));
  
  B.rows = dimension;
  B.cols = dimension;
  B.matrix = malloc((B.rows*B.cols)*sizeof(_MTYPE));
  
  C.rows = A.rows;
  C.cols = B.cols;
  C.matrix = calloc(C.rows*C.cols,sizeof(_MTYPE));
  
  
  for(long i=0; i<A.rows;i++)
  {
    for(long j=0;j<A.cols;j++)
    {
      if(i==j)
	      A.matrix[j+(i*A.cols)] = 5.0;
      else
	      A.matrix[j+(i*A.cols)] = 4.0;
    }
  }
  
  for(long i=0; i<B.rows;i++)
  {
    for(long j=0;j<B.cols;j++)
    {
      if(i==j)
        B.matrix[j+(i*B.cols)] = 1.0;   
      else
	      B.matrix[j+(i*B.cols)] = 0.0;
    }
  }
  
  
  sec = cclock();
  
  multiplication(A,B,C);	
  
  sec = cclock() -sec;
  
  double gflops = ((double) C.cols)*((double) C.rows)*((double) A.cols)*2.0;
  
  gflops = (gflops/sec)/10e9;
  printf("time = %5.2G sec \n",sec);
  printf("flops = %5.2E GFLOPS \n",gflops);
  
  //check results
  int check = 0;
  for(long i=0; i<C.rows;i++){
    for(long j=0;j<C.cols;j++){
      if(i==j & C.matrix[j+(i*A.cols)] != 5.0){
	check = 1;
	printf("ERROR %lf \n",C.matrix[j+(i*A.cols)]);}
	else if(i!=j & C.matrix[j+(i*A.cols)] != 4.0){
	  check = 1;
	  printf("ERROR %lf \n",C.matrix[j+(i*A.cols)]);}
    }
  }
  
  if(check == 0)
    printf("\n");
  else
    printf("MATRIX CALC ERROR\n");
  
  free(A.matrix);
  free(B.matrix);
  free(C.matrix);
  
  return 0;
  
}
