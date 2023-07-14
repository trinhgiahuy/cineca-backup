#include <stdio.h>

__global__
void doLoop()
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d\n", i);
}

int main()
{
  // Other possible kernel configurations are
  // <<<5, 2>>>
  // <<<10, 1>>>
  doLoop<<<2, 5>>>();
  cudaDeviceSynchronize();
 }