/**
 * reduce_sum
 * baseline: 串行处理
 * latency: 1027.339844 ms
 */
#include <cuda.h>
#include <iostream>

// 错误检查宏
#define cudaCheckError() {                                      \
    cudaError_t e = cudaGetLastError();                         \
    if (e != cudaSuccess) {                                     \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(e));                         \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

__global__ void reduce_baseline(const int* input, int* output, size_t n) {
    // 由于只分配了1个block和thread,此时cuda程序相当于串行程序
    int sum = 0;
    for(size_t i = 0; i < n; ++i) {
        sum += input[i];
    }
    *output = sum;
}

bool checkResult(int* out, int groudtruth, int n) {
    if (*out != groudtruth) {
        return false;
    }
    return true;
}

int main() {
    // const int N = 32 * 1024 * 1024;
    const int N = 25600000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int blockSize = 1;
    int gridSize = 1;

    int *a = (int *)malloc(N * sizeof(int));
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));

    int *out = (int *)malloc(gridSize * sizeof(int));  // 为啥这里是gridSize(每个block内的线程reduce到一起吗)
    int *d_out;
    cudaMalloc((void **)&d_out, gridSize * sizeof(int));

    for(int i = 0; i < N; i++) {
        a[i] = 1;
    }
    int groudtruth = N * 1;
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    // 定义block数量和threads的数量
    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_baseline<<<1, 1>>>(d_a, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaCheckError();
    cudaMemcpy(out, d_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = checkResult(out, groudtruth, gridSize);
    
    if(is_right) {
        printf("the ans is right!\n");
    }
    else {
        printf("the ans is wrong!\n");
        for(int i = 0; i < gridSize;i++){
            printf("res per block : %1f ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_baseline latency = %f ms\n", milliseconds);


    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}