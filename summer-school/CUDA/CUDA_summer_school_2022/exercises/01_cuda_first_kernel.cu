#include <stdio.h>

void onCPU() {
    printf("This function runs on CPU\n");
    }

__global__
void onGPU() {
    printf("This function runs on GPU\n");
    }

int main() {
    onCPU();

    onGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
}