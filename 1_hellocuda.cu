#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// __global__修饰的内核函数
__global__ void hello_cuda(){
    // 泛指当前线程在所有block范围内的全局id
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("block id = [ %d ], thread id = [ %d ] hello cuda\n", blockIdx.x, idx);
}

int main() {
    hello_cuda<<< 2, 3 >>>();  //<<<>>>是启动内核函数的标志
    cudaDeviceSynchronize();  //主机和GPU异步执行，这步为cpu等待GPU执行完毕
    return 0;  
}