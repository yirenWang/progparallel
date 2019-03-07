#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <float.h>

using namespace std;

__global__ void gpu_saxpy(int n, float *x, float *y, float *s)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
     s[i] = ( x[i] + y[i] ) / 2;
}

void cpu_saxpy(int n, float *x, float *y, float *s)
{
  #pragma omp parallel  for num_threads(8) 
  for (int i=0; i<n; i++)
  {
    s[i] = ( x[i] + y[i] ) / 2;
  }
}

void cpu_saxpy_mono(int n, float *x, float *y, float *s)
{
  for (int i=0; i<n; i++)
  {
    s[i] = ( x[i] + y[i] ) / 2;
  }
}


int main(void)
{
  unsigned long int N = 4096*4096;
  float *x, *y, *s_cpu, *s_gpu, *d_x, *d_y, *d_s;
  
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  s_gpu = (float*)malloc(N*sizeof(float));
  s_cpu = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_s, N*sizeof(float));

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
  cpu_saxpy_mono(N, x, y, s_cpu);  

  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "cpu time mono \t" << std::chrono::duration<double>(t1 - t0).count() << std::endl;

  // Perform SAXPY on 16M elements (GPU)
  for(int k = 16; k < 1024; k+=8)
  {
    std::cout << "k = " << k << std::endl;

    t0 = std::chrono::high_resolution_clock::now();
  
    gpu_saxpy<<<(N+k)/k, k>>>(N, d_x, d_y, d_s);  
    
    t1 = std::chrono::high_resolution_clock::now();
  
    std::cout << "gpu time \t" << std::chrono::duration<double>(t1 - t0).count() << std::endl;
  
  }

  t0 = std::chrono::high_resolution_clock::now();

  // Perform SAXPY on 16M elements (CPU)
  cpu_saxpy(N, x, y, s_cpu);  

  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "cpu time para \t" << std::chrono::duration<double>(t1 - t0).count() << std::endl;

  cudaMemcpy(s_gpu, d_s, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::max(maxError, s_cpu[i]-s_gpu[i]);
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_s);
  free(x);
  free(y);
  free(s_cpu);
  free(s_gpu);
}

