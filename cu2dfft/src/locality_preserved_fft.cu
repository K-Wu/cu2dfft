// code from https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf
#include "cu2dfft.h"

const int BSZ = 4;
const int N = 128*128;

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif



int main()
{
    float xmax = 1.0f, xmin = 0.0f, ymin = 0.0f,
          h = (xmax - xmin) / ((float)N), s = 0.1, s2 = s * s;
    float *x = new float[N * N], *y = new float[N * N], *u = new float[N * N],
          *f = new float[N * N], *u_a = new float[N * N], *err = new float[N * N];
    float r2;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
        {
            x[N * j + i] = xmin + i * h;
            y[N * j + i] = ymin + j * h;
            r2 = (x[N * j + i] - 0.5) * (x[N * j + i] - 0.5) + (y[N * j + i] - 0.5) * (y[N * j + i] - 0.5);
            f[N * j + i] = (r2 - 2 * s2) / (s2 * s2) * exp(-r2 / (2 * s2));
            u_a[N * j + i] = exp(-r2 / (2 * s2)); // analytical solution
        }
    float *k = new float[N];
    for (int i = 0; i <= N / 2; i++)
    {
        k[i] = i * 2 * M_PI;
    }
    for (int i = N / 2 + 1; i < N; i++)
    {
        k[i] = (i - N) * 2 * M_PI;
    }
    // Allocate arrays on the device
    float *k_d, *f_d, *u_d;
    cudaMalloc((void **)&k_d, sizeof(float) * N);
    cudaMalloc((void **)&f_d, sizeof(float) * N * N);
    cudaMalloc((void **)&u_d, sizeof(float) * N * N);
    cudaMemcpy(k_d, k, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(f_d, f, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cufftComplex *ft_d, *f_dc, *ft_d_k, *u_dc;
    cudaMalloc((void **)&ft_d, sizeof(cufftComplex) * N * N);
    cudaMalloc((void **)&ft_d_k, sizeof(cufftComplex) * N * N);
    cudaMalloc((void **)&f_dc, sizeof(cufftComplex) * N * N);
    cudaMalloc((void **)&u_dc, sizeof(cufftComplex) * N * N);
    dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
    dim3 dimBlock(BSZ, BSZ);
    real2complex<<<dimGrid, dimBlock>>>(f_d, f_dc, N);
    //cufftHandle plan;
    //cufftPlan2d(&plan, N, N, CUFFT_C2C);
    //cufftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);
    
    

    //cufftExecC2C(plan, ft_d_k, u_dc, CUFFT_INVERSE);
    mycufftHandle plan;
    mycufftPlan1d(&plan, 1024*1024*16, CUFFT_C2C, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    mycu1dfftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout<<"Time: "<<elapsed<<" ms"<<std::endl;


    complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);
    cudaMemcpy(u, u_d, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
    float constant = u[0];
    for (int i = 0; i < N * N; i++)
    {
        u[i] -= constant; // substract u[0] to force the arbitrary constant to be 0
    }
}