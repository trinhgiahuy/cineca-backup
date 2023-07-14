#include <array>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
constexpr int N = 128;

// Exercise 2:
// Offload this code to device using single_task

int main() {

    queue q;

    // Memory is ready for SYCL
    // No need to change this for this exercise
    int* A = malloc_shared<int>(N*N, q);
    int* B = malloc_shared<int>(N*N, q);
    int* C = malloc_shared<int>(N*N, q);

    for (int i = 0; i < N; ++i) {
        A[i*(N+1)] = 1;
        B[i*(N+1)] = i;
    }

	// This loop is serial
	q.submit([&](handler& h){
		h.single_task([=](){
			for (int i = 0; i < N; i++){
				for (int j = 0; j < N; j++) {
					for (int k = 0; k < N; k++) {
						C[i*N + j] += A[i*N + k] * B[k*N + j];
					}
				}
            }
        });
	}).wait();

    // Expected 0 N-1
    std::cout << C[0] << " " << C[N*N-1] << std::endl;

    // Always free memory at the end
    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}
