#include <stdio.h>

void init(int *a, int N) {
  int i;
  for (i = 0; i < N; ++i)
    a[i] = i;
}

__global__
void doubleElements(int *a, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
    a[i] *= 2;
}

bool checkElementsAreDoubled(int *a, int N) {
  int i;
  for (i = 0; i < N; ++i) {
    if (a[i] != i*2)
      return false;
  }
  return true;
}

int main()
{
  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  size_t threads_per_block = 1024;
  size_t number_of_blocks = 32;

  cudaError_t syncErr, asyncErr;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

  syncErr = cudaGetLastError();
  asyncErr = cudaDeviceSynchronize();

  if (syncErr != cudaSuccess) printf("ErrorSync: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("ErrorAsync: %s\n", cudaGetErrorString(asyncErr));

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
  