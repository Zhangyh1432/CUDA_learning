#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

typedef float FLOAT;

/* CUDA kernel function */
__global__ void vec_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    /* 2D grid */
    static int i = 0;

    if (i == 0)
    {
        // printf("gridDim.x %d\n", gridDim.x); // 7
        // printf("gridDim.y %d\n", gridDim.y); // 7
        // printf("gridDim.z %d\n", gridDim.z); // 1
        // printf("blockDim.x %d\n", blockDim.x);   // 256
        // printf("blockDim.y %d\n", blockDim.y);   // 1
        // printf("blockDim.z %d\n", blockDim.z);   // 1
    }

    i++;
    int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);

    /* 1D grid */
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N)
        z[idx] = y[idx] + x[idx];
}

void vec_add_cpu(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    for (int i = 0; i < N; i++)
        z[i] = y[i] + x[i];
}

int main()
{
    int N = 10000;
    int nbytes = N * sizeof(FLOAT);

    /* 1D block */
    int bs = 256;

    /* 2D grid */
    int s = ceil(sqrt((N + bs - 1.) / bs));
    cout << "s = " << s << endl;
    dim3 grid(s, s);
    /* 1D grid */
    // int s = ceil((N + bs - 1.) / bs);
    // dim3 grid(s);

    FLOAT *dx, *hx;
    FLOAT *dy, *hy;
    FLOAT *dz, *hz;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);

    /* init time */
    float milliseconds = 0;

    /* alllocate CPU mem */
    hx = (FLOAT *)malloc(nbytes);
    hy = (FLOAT *)malloc(nbytes);
    hz = (FLOAT *)malloc(nbytes);

    /* init */
    for (int i = 0; i < N; i++)
    {
        hx[i] = 1;
        hy[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    /* launch GPU kernel */
    vec_add<<<grid, bs>>>(dx, dy, dz, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    /* copy GPU result to CPU */
    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

    /* CPU compute */
    FLOAT *hz_cpu_res = (FLOAT *)malloc(nbytes);
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    /* check GPU result with CPU*/
    for (int i = 0; i < N; ++i)
    {
        if (fabs(hz_cpu_res[i] - hz[i]) > 1e-6)
        {
            cout << "Result verification failed at element index " << i << endl;
        }
    }
    cout << "Result right\n";
    cout << "Mem BW=" << ((float)N * 4 / milliseconds / 1e6) << " (GB/sec)\n"; /// 1.78gb/s
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);

    return 0;
}