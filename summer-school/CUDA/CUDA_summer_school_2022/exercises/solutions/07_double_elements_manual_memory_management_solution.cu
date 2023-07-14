#include <stdio.h>

// Initialize array values on the host.
void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
    a[i] = i;
}

// Double elements in parallel on the GPU.
__global__
void doubleElements(int *a, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    a[i] *= 2;
}

// Check all elements have been doubled on the host.
bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i) {
    if (a[i] != i*2)
      return false;
  }
  return true;
}

int main()
{
  int N = 100;
  int *device_a, *host_a;

  size_t size = N * sizeof(int);

  cudaMalloc(&device_a, size);
  cudaMallocHost(&host_a, size);

  init(host_a, N);

  cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);  
  
  size_t threads_per_block = 10;
  size_t number_of_blocks = 10;

  doubleElements<<<number_of_blocks, threads_per_block>>>(device_a, N);

  cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);

  bool areDoubled = checkElementsAreDoubled(host_a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(device_a);
  cudaFreeHost(host_a);
 }
  