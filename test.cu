
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello()
{
    printf("Hello from CUDA thread %d\n",
           threadIdx.x);
}

int main()
{
    hello<<<1, 5>>>();

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        printf("Error: %s\n",
               cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA execution successful!\n");
    return 0;
}

