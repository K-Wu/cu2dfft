#pragma once
#include "cu2dfft.h"
#include "thrust_helpers.h"
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

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#ifndef M_PIf32
#define M_PIf32 3.1415926535897932384626433832795f
#endif 