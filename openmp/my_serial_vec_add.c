#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>
//#include <iostream>
//#include <fstream>
//#include <string>

int main(int argc, char *argv[])
{
  //do something
  int i;
  int n = 10;
  double tick, tock;

  float *a;
  float *b;
  float *c;

  printf("\n");
  printf("Vector addition\n");
  printf("  C/OpenMP version\n");
  printf("\n");
  printf("  A program which adds two vector.\n");

  /*
    Allocate the vector data.
  */
  a = (float *)malloc(n * sizeof(float));
  if (a == NULL) {
    printf("Failed to allocate memory for array a\n");
    return 1; // or handle the error accordingly
  }

  b = (float *)malloc(n * sizeof(float));
  if (b == NULL) {
    printf("Failed to allocate memory for array a\n");
    return 1; // or handle the error accordingly
  }

  c = (float *)malloc(n * sizeof(float));
  if (c == NULL) {
    printf("Failed to allocate memory for array a\n");
    return 1; // or handle the error accordingly
  }

  for (i = 0; i < n; i++)
  {
    a[i] = (float)(1);
    b[i] = (float)(2);
    c[i] = (float)(0);
  }

  // Copy value to device
  #pragma omp target update to(a[0:n],b[0:n])
  {}

  bool USE_OPENMP_CLOCK = true;

  omp_set_num_threads(8);
  tick = USE_OPENMP_CLOCK ? omp_get_wtime() : clock();
  
  #pragma omp parallel for firstprivate(c)
  //#pragma omp target teams distribute parallel for
  for (i = 0; i < n; i++)
  {
    auto tid = omp_get_thread_num();
    //printf("[Before]Thread id %d have c[i]=%f\n",tid,c[i]);
    c[i] = a[i] + b[i];
    //printf("[After]Thread id %d have c[i]=%f\n",tid,c[i]);
  }
  printf("c after loop: %10.4f",c[10]);
  #pragma omp target update from(c[0:n])
  {}

  tock = USE_OPENMP_CLOCK ? omp_get_wtime() : clock(); 

  /*
    Print a few entries.
  */
  printf("\n");
  printf("   i        a[i]        b[i]      c[i] = a[i] + b[i]\n");
  printf("\n");
  for (i = 0; i < n && i < 10; i++)
  {
    printf("  %2d  %10.4f  %10.4f  %10.4f\n", i, a[i], b[i], c[i]);
  }
  /*
    Free memory.
  */
  free(a);
  free(b);
  free(c);
  /*
    Terminate.
  */
  printf("\n");
  printf("Vector addition\n");
  printf("  Normal end of execution.\n");

  printf("===================================== \n");
  printf("Work took %f seconds\n", (tock - tick) );
  printf("===================================== \n");
  
  //Write result to file
  //std::string fn="openmp.log";
  //std::ofstream outputFile;
  //outputFile.open(fn,std::ios::app);
  //outputFile<<(tock-tick)<<std::endl;

  //outputFile.close();
  return 0;
}
