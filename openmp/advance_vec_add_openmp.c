#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void die(const char *message, const int line, const char *file);

void initialise(float **a_ptr, float **b_ptr, float **c_ptr, const int N);
void finalise(float **a_ptr, float **b_ptr, float **c_ptr, const int N);

int main(int argc, char const *argv[]) {
  int N = 10; /* vector size */
  float *a = NULL;
  float *b = NULL;
  float *c = NULL;
  double tick,tock;
  initialise(&a, &b, &c, N);

  // Set values of a and b on the host
  for (int i = 0; i < N; i++) {
    a[i] = 1.f;
    b[i] = 2.f;
  }

  // Copy the values to the device
  #pragma omp target update to(a[0:N], b[0:N])
  {}

  // Execute vecadd on the target device
  tick = omp_get_wtime();
  //#pragma omp target teams distribute parallel for

  #pragma omp parallel for private(c)
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];

  }
  tock = omp_get_wtime();

  // Copy the result from the device
  #pragma omp target update from(c[0:N])
  {}

  
  printf("Work took %f seconds\n", (tock-tick));
  // Verify the results
  int correct_results = 1;
  for (int i = 0; i < N; i++) {
    if (fabs(c[i] - 3.f) > 0.00001f) {
      printf("Incorrect answer at index %d\n", i);
      correct_results = 0;
    }
  }

  if (correct_results) {
    printf("Success!\n");
  }

  finalise(&a, &b, &c, N);
  return 0;
}

//void initialise(float **a_ptr, float **b_ptr, float **c_ptr, const int N) {

//}

void initialise(float **a_ptr, float **b_ptr, float **c_ptr, const int N) {
  *a_ptr = malloc(sizeof(float) * N);
  if (*a_ptr == NULL)
    die("cannot allocate memory for a", __LINE__, __FILE__);
  *b_ptr = malloc(sizeof(float) * N);
  if (*b_ptr == NULL)
    die("cannot allocate memory for b", __LINE__, __FILE__);
  *c_ptr = malloc(sizeof(float) * N);
  if (*c_ptr == NULL)
    die("cannot allocate memory for c", __LINE__, __FILE__);

  // Set up data region on device
  #pragma omp target enter data map(alloc: (*a_ptr)[0:N], (*b_ptr)[0:N], (*c_ptr)[0:N])
  {}
}

void finalise(float **a_ptr, float **b_ptr, float **c_ptr, const int N) {
  // Have to place all pointers into local variables
  // for OpenMP to accept them in mapping clauses
 // float *a = *a_ptr;
  //float *b = *b_ptr;
  //float *c = *c_ptr;

  // End data region on device
  #pragma omp target exit data map(release: (*a_ptr)[0:N], (*b_ptr)[0:N], (*c_ptr)[0:N])
  {}

  free(*a_ptr);
  *a_ptr = NULL;
  free(*b_ptr);
  *b_ptr = NULL;
  free(*c_ptr);
  *c_ptr = NULL;
}

void die(const char *message, const int line, const char *file) {
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}
