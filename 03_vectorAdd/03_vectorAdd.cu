 /**
 * 向量加法：z = x + y
 */

#include <cstdio>
#include <cuda.h>
#include <time.h>

// 错误检查宏
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                \
    if(e!=cudaSuccess) {                                             \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,         \
               cudaGetErrorString(e));                               \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

__global__ void vec_add(float *x, float *y, float *z, int N) {

    // 1D
    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 2D
    size_t idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if(idx < N) z[idx] = x[idx] + y[idx];
}


void vec_add_cpu(float *x, float *y, float *z, int N) {
    for(int i = 0; i < N; i++) {
        z[i] = x[i] + y[i];
    }
}

void check_result(float *x, float *y, int N) {
    for(int i = 0; i < N; i++) {
        if(fabs(x[i] - y[i]) > 1e-6) {  // 整数绝对值为abs(), 浮点数绝对值为fabs()
            printf("Result verification failed at element index %d!\n", i);
        }
    }
    printf("Result check right!\n");
}


int main() {
    int N = 1000000;
    int nbytes = N * sizeof(float);

    float *hx, *hy, *hz;
    hx = (float*) malloc(nbytes);
    hy = (float*) malloc(nbytes);
    hz = (float*) malloc(nbytes);
    if (hx == NULL || hy == NULL || hz == NULL) {
        printf("Failed to allocate host vectors.\n");
        return -1;
    }
    for(int i = 0; i < N; i++) {
        hx[i] = 1.0f;
        hy[i] = 1.0f;
    }
    float *hz_cpu_res = (float *) malloc(nbytes);

    clock_t s, e;
    s = clock();
    vec_add_cpu(hx, hy, hz_cpu_res, N);
    e = clock();
    double milliseconds_cpu = ((double) (e - s)) * 1000.0 / CLOCKS_PER_SEC;

    float *dx, *dy, *dz;
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes); 
    cudaMalloc((void **)&dz, nbytes);
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    // int grid = (N + block_size - 1) / 256;
    int grid_size = ceil(sqrt((N + block_size - 1.) / block_size));
    dim3 grid(grid_size, grid_size);  // 因为是二维，所以传入二维 grid_size 即 dim3 grid
    
    float milliseconds_cuda = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vec_add<<<grid, block_size>>>(dx, dy, dz, N);  // 多数CUDA设备的最大线程数是1024
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_cuda, start, stop); 
    
    cudaCheckError();
    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);
    check_result(hz, hz_cpu_res, N);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);

    printf("===>\n");
    printf("vec_add in cpu spend %f\n", milliseconds_cpu);
    printf("vec_add in cuda(2D) spend %g\n", milliseconds_cuda);
    return 0; 
}

