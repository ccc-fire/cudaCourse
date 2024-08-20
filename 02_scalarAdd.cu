#include <cstdio>
#include <cuda.h>

__global__ void sum_cuda(float* x) {
    // size_t block_id = blockIdx.x;
    // size_t local_idx = threadIdx.x;
    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("block=%d, thread_id in block=%d, thread_id in global=%d\n", block_id, local_idx, global_idx);
    x[global_idx] += 10;
}

int main() {
    int N = 12;
    int nbytes = N * sizeof(float);

    float* hx;
    hx = (float*) malloc(nbytes);
    printf("hx original: \n");
    for(int i = 0; i < N; i++) {
        hx[i] = i;
        printf("%g\t", hx[i]);
    }
    printf("\n");


    float* dx;
    cudaMalloc((void**)&dx, nbytes);
    // 二级指针：为了传递指针的地址，为了修改dx的值，使其指向设备内存
    // 因为dx初始化的时候没有指向设备内存（被初始化为空（未指向任何有效内存））
    // 因此需要使用二级指针传递 dx 的地址以修改 dx
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    sum_cuda<<<1, N>>>(dx);
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    printf("hx current: \n");
    for(int i = 0; i < N; i++) {
        printf("%g\t", hx[i]);
    }

    // 释放内存
    free(hx);
    cudaFree(dx);

    cudaDeviceSynchronize();
    printf("\nFinish!\n");
    return 0;
}