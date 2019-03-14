#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"

#ifndef SAXPY_CUH
#define SAXPY_CUH

__global__ void gpu_saxpy(int n, float a, float *x, float *y, float *s);
__global__ void gpu_sobel(u_char *Source, u_char *Resultat, unsigned int height, unsigned int width);
__global__ void gpu_sobel_shared(u_char *Source, u_char *Resultat, unsigned int height, unsigned int width);

#endif