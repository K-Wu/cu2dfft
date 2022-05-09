#pragma once
#include "cu2dfft.h"
__global__ void real2complex(float *f, cufftComplex *fc, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * N + i;
    if (i < N && j < N)
    {
        fc[index].x = f[index];
        fc[index].y = 0.0f;
    }
}
__global__ void complex2real(cufftComplex *fc, float *f, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = j * N + i;
    if (i < N && j < N)
    {
        f[index] = fc[index].x / ((float)N * (float)N);
        // divide by number of elements to recover value
    }
}