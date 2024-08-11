#include <cstdio>
#include <cuda.h>

__global__ void hello_cuda() {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("[ %d ]: hello cuda!\n", idx);
}

int main() {
    hello_cuda<<<1, 14>>>();
    cudaDeviceSynchronize();
    printf("hello world!\n");
}
