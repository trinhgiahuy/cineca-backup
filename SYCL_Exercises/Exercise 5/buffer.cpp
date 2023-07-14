#include <array>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
constexpr int N = 128;

// Exercise 5:
// Using results of exercise 3 allocate memory using buffers

int main() {

    queue q;

    int* A = new int[N*N];
    int* B = new int[N*N];
    int* C = new int[N*N];

    for (int i = 0; i < N*N; i++) {
        A[i] = B[i] = C[i] = 0;
    }

    for (int i = 0; i < N; ++i) {
        A[i*(N+1)] = 1;
        B[i*(N+1)] = i;
    }

    // Hint:
    // Remember to use accessors inside the submit

    // Parallelize this code (Exercise 3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }

    // Hint:
    // Remember to use host_accessor to show results

    // Expected 0 N-1
    std::cout << C[0] << " " << C[N*N-1] << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
