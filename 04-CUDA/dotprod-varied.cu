#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <float.h>

using namespace std;

__global__ void gpu_saxpy(int n, float *x, float *y, float* s)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    atomicAdd(s, x[i] * y[i]);

}

void cpu_saxpy(int n, float *x, float *y, float s)
{
  #pragma omp parallel for reduction(+ : s)
  for (unsigned long i = 0; i < n; i++)
  {
    s += x[i] * y[i];
  }
}

int cpu_saxpy_mono(int n, float *x, float *y, float s)
{
  for (int i=0; i<n; i++)
  {
    s += x[i] * y[i];
  }
  return 0;
}


int main(void)
{
  for (int n = 1; n < 4096; n*=2)
  {
    int N = n*n;
    std::cout << "N=" << N << std::endl; 

    float *x, *y, s_cpu, *s_gpu, *d_x, *d_y, *d_s;
  
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));
    s_gpu = (float*)malloc(1*sizeof(float));
    s_cpu = 0;
    cudaMalloc(&d_x, N*sizeof(float)); 
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMalloc(&d_s, 1*sizeof(float));
  
    for (int i = 0; i < N; i++) {
      x[i] = 1.0f;
      y[i] = 2.0f;
    }
  
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  
    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  
    t0 = std::chrono::high_resolution_clock::now();
  
    // Perform SAXPY on 16M elements (CPU)
    int a = cpu_saxpy_mono(N, x, y, s_cpu);  
    std::cout << a << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
  
    std::cout << "cpu time mono \t" << std::chrono::duration<double>(t1 - t0).count() << std::endl;
  
    *s_gpu = 0;

    std::chrono::high_resolution_clock::time_point t_mc0 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_s, s_gpu, 1*sizeof(float), cudaMemcpyHostToDevice);
    std::chrono::high_resolution_clock::time_point t_mc1 = std::chrono::high_resolution_clock::now();

    t0 = std::chrono::high_resolution_clock::now();
    
    int k = 32;
    gpu_saxpy<<<(N+k)/k, k>>>(N, d_x, d_y, d_s);  
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
  
    std::cout << "gpu time \t" << std::chrono::duration<double>(t1 - t0).count() << std::endl;
  
    t0 = std::chrono::high_resolution_clock::now();
  
    // Perform SAXPY on 16M elements (CPU)
    cpu_saxpy(N, x, y, s_cpu);  
  
    t1 = std::chrono::high_resolution_clock::now();
  
    std::cout << "cpu time para \t" << std::chrono::duration<double>(t1 - t0).count() << std::endl;

    std::chrono::high_resolution_clock::time_point t_mc2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(s_gpu, d_s, 1*sizeof(float), cudaMemcpyDeviceToHost);
    std::chrono::high_resolution_clock::time_point t_mc3 = std::chrono::high_resolution_clock::now();
    
    std::cout << "gpu mem copy time \t" << std::chrono::duration<double>(t_mc3-t_mc2 + t_mc1 - t_mc0).count() << std::endl;

    float maxError = 0.0f;
    maxError = std::max(maxError, s_cpu-*s_gpu);
    printf("Max error: %f\n", maxError);
  
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_s);
    free(x);
    free(y);
    free(s_gpu);  
  }
}

