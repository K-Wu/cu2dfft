// code from https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf
#include "cu2dfft.h"

const int BSZ = 4;
//const int N = 16*1024;
const int N = 64;


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
    cuda_err_chk(cudaMalloc((void **)&k_d, sizeof(float) * N));
    cuda_err_chk(cudaMalloc((void **)&f_d, sizeof(float) * N * N));
    cuda_err_chk(cudaMalloc((void **)&u_d, sizeof(float) * N * N));
    cuda_err_chk(cudaMemcpy(k_d, k, sizeof(float) * N, cudaMemcpyHostToDevice));
    cuda_err_chk(cudaMemcpy(f_d, f, sizeof(float) * N * N, cudaMemcpyHostToDevice));
    cufftComplex *ft_d, *f_dc;
    cufftComplex *my_f_dc, *my_ft_d;
    cufftComplex *ft_d_k, *u_dc;

    cuda_err_chk(cudaMalloc((void **)&ft_d_k, sizeof(cufftComplex) * N * N));
    //cuda_err_chk(cudaMalloc((void **)&ft_d, sizeof(cufftComplex) * N * N));
    //cuda_err_chk(cudaMalloc((void **)&f_dc, sizeof(cufftComplex) * N * N));

    //cuda_err_chk(cudaMalloc((void **)&my_ft_d, sizeof(cufftComplex) * N * N));
    //cuda_err_chk(cudaMalloc((void **)&my_f_dc, sizeof(cufftComplex) * N * N));

    cuda_err_chk(cudaMalloc((void **)&u_dc, sizeof(cufftComplex) * N * N));

    thrust::device_vector<cufftComplex> ft_d_v(N * N, make_float2(0.0f, 0.0f));
    thrust::device_vector<cufftComplex> f_dc_v(N * N, make_float2(0.0f, 0.0f));
    thrust::device_vector<cufftComplex> my_ft_d_v(N * N, make_float2(0.0f, 0.0f));
    thrust::device_vector<cufftComplex> my_f_dc_v(N * N, make_float2(0.0f, 0.0f));
    ft_d = thrust::raw_pointer_cast(ft_d_v.data());
    f_dc = thrust::raw_pointer_cast(f_dc_v.data());
    my_ft_d = thrust::raw_pointer_cast(my_ft_d_v.data());
    //my_f_dc = thrust::raw_pointer_cast(my_f_dc_v.data());

    dim3 dimGrid(int((N - 0.5) / BSZ) + 1, int((N - 0.5) / BSZ) + 1);
    dim3 dimBlock(BSZ, BSZ);
    real2complex<<<dimGrid, dimBlock>>>(f_d, f_dc, N);
    //cuda_err_chk(cudaMemcpy(my_f_dc, f_dc, N * N * sizeof(cufftComplex), cudaMemcpyDeviceToDevice));
    cufftHandle plan;
    //cufftPlan2d(&plan, N, N, CUFFT_C2C);
    cufftPlan1d(&plan, N*N/*1024*1024*128*/, CUFFT_C2C,1);
// 
    cufftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);
// 
    cufftExecC2C(plan, ft_d_k, u_dc, CUFFT_INVERSE);
    mycufftHandle myplan;
    mycufftPlan1d(&myplan, N*N/*1024*1024*128*/, CUFFT_C2C, 1);
    cuda_err_chk(cudaDeviceSynchronize());
    for (int idx = 0; idx < 10; idx++)
    {

        cudaEvent_t start, stop;
        cuda_err_chk(cudaEventCreate(&start));
        cuda_err_chk(cudaEventCreate(&stop));
        cuda_err_chk(cudaEventRecord(start, 0));
        mycu1dfftExecC2C(myplan, f_dc, my_ft_d, CUFFT_FORWARD);
        cuda_err_chk(cudaEventRecord(stop, 0));
        cuda_err_chk(cudaEventSynchronize(stop));
        float elapsed;
        cuda_err_chk(cudaEventElapsedTime(&elapsed, start, stop));
        std::cout << "Time: " << elapsed << " ms" << std::endl;
    }

    cuda_err_chk(cudaDeviceSynchronize());

    std::cout << thrust::equal(thrust::device, ft_d_v.begin(), ft_d_v.end(), my_ft_d_v.begin(), is_close_float2()) << std::endl;
    std::cout << thrust::equal(thrust::device, my_ft_d_v.begin(), my_ft_d_v.end(), ft_d_v.begin(), is_close_float2()) << std::endl;

    print_range("my_ft_d_v", my_ft_d_v.begin(), my_ft_d_v.end());

    print_range("ft_d_v", ft_d_v.begin(), ft_d_v.end());

    complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);
    cuda_err_chk(cudaMemcpy(u, u_d, sizeof(float) * N * N, cudaMemcpyDeviceToHost));
    float constant = u[0];
    for (int i = 0; i < N * N; i++)
    {
        u[i] -= constant; // substract u[0] to force the arbitrary constant to be 0
    }
    //cuda_err_chk(cudaFree(ft_d));
    //cuda_err_chk(cudaFree(my_ft_d));
    //cuda_err_chk(cudaFree(f_dc));
    cuda_err_chk(cudaFree(ft_d_k));
    cuda_err_chk(cudaFree(u_dc));
    cuda_err_chk(cudaFree(k_d));
    cuda_err_chk(cudaFree(f_d));
    cuda_err_chk(cudaFree(u_d));
}