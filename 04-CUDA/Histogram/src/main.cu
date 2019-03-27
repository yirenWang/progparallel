#include <stdio.h>
#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"

#include "gpu_sobel.cuh"

#include "container.cuh"
#include "ios.cuh"

#include "parameters.cuh"


#define DEBUG 1

void cpu_sobel(u_char **Source, unsigned long long *Resultat, unsigned int height, unsigned int width, unsigned int nthreads) {

    for (auto i = 0; i < height; i++) {
        for (auto j = 0; j < width; j++) {
            Resultat[(unsigned int)Source[i][j]]  += 1;
        }
    }
}

int main(int argc, char* argv[]) {
    unsigned int height, width;
    unsigned int nthreads = omp_get_max_threads();

    std::string image_filename(argv[1]);
    int ITER =  atoi(argv[2]);

    get_source_params(image_filename, &height, &width);
    std::cout << width << " " << height << std::endl;
    u_char **Source, *d_Source;
    unsigned long long *Resultat, *ResultatGPU, *ResultatGPUShared, *d_ResultatGPUShared, *d_ResultatGPU;


	image<u_char> imgSource(height, width, &Source);
    Resultat = (unsigned long long *)malloc(256*sizeof(unsigned long long));    
    ResultatGPU = (unsigned long long *)malloc(256*sizeof(unsigned long long));    
    ResultatGPUShared = (unsigned long long *)malloc(256*sizeof(unsigned long long));    
       
    auto fail = init_source_image(Source, image_filename, height, width);
    if (fail) {
        std::cout << "Chargement impossible de l'image" << std::endl;
        return 0;
    }
 
    cudaMalloc(&d_Source, height*width*sizeof(u_char));    
    cudaMalloc(&d_ResultatGPU, 256*sizeof(unsigned long long));    
    cudaMalloc(&d_ResultatGPUShared, 256*sizeof(unsigned long long));    

    std::chrono::high_resolution_clock::time_point t_mc0 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_Source, Source[0], height*width*sizeof(u_char), cudaMemcpyHostToDevice);

    std::chrono::high_resolution_clock::time_point t_mc1 = std::chrono::high_resolution_clock::now();


    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

    for (auto it =0; it < ITER; it++) {
        cpu_sobel(Source, Resultat, height, width, nthreads);
    }

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration<double>(t1-t0).count()/ITER;

    dim3 threads(BLOCKDIM_X,BLOCKDIM_Y);
    dim3 blocks(width/BLOCKDIM_X + 1,height/BLOCKDIM_Y + 1);
    t0 = std::chrono::high_resolution_clock::now();
    for (auto it =0; it < ITER; it++) {
        gpu_sobel<<<blocks,threads>>>(d_Source, d_ResultatGPU, height, width);
        cudaDeviceSynchronize();
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration<double>(t1-t0).count()/ITER;

    std::chrono::high_resolution_clock::time_point t_mc2 = std::chrono::high_resolution_clock::now();

    cudaMemcpy(ResultatGPU, d_ResultatGPU, 256*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    std::chrono::high_resolution_clock::time_point t_mc3 = std::chrono::high_resolution_clock::now();

    auto gpu_mem_copy_duration = std::chrono::duration<double>(t_mc3-t_mc2 + t_mc1 - t_mc0).count()/ITER;

    dim3 blocks2(width/(BLOCKDIM_X)+1,height/(BLOCKDIM_Y)+1);
    dim3 threads2(16,16);
    t0 = std::chrono::high_resolution_clock::now();
    for (auto it =0; it < ITER; it++) {
        gpu_sobel_shared<<<blocks2,threads2>>>(d_Source, d_ResultatGPUShared, height, width);
        cudaDeviceSynchronize();
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto gpu_duration_shared = std::chrono::duration<double>(t1-t0).count()/ITER;

    cudaMemcpy(ResultatGPUShared, d_ResultatGPUShared, 256*sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << cpu_duration << " "  << gpu_duration << " "  << gpu_duration_shared << " " << gpu_mem_copy_duration << std::endl;
    std::cout << ResultatGPU[0] << " " << ResultatGPU[255] << std::endl;
    std::cout << ResultatGPUShared[0] << " " << ResultatGPUShared[255] << std::endl;
    std::cout << Resultat[0] << " " << Resultat[255] << std::endl;
    
    return 0;
}

