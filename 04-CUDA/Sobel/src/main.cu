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

void cpu_sobel(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width, unsigned int nthreads) {

    for (auto i = 1; i < height-1; i++) {
        for (auto j = 1; j < width-1; j++) {
            if ((i==0)||(i==height-1)||(j==0)||(j==width-1)) {Resultat[i][j]=0;}
            else {
                Resultat[i][j]  = std::abs(Source[i-1][j-1] + Source[i-1][j] + Source[i-1][j+1] - (Source[i+1][j-1] + Source[i+1][j] + Source[i+1][j+1]));
                Resultat[i][j] += std::abs(Source[i-1][j-1] + Source[i][j-1] + Source[i+1][j-1] - (Source[i-1][j+1] + Source[i][j+1] + Source[i+1][j+1]));
            }
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
    u_char **Source, **Resultat, **ResultatGPU, **ResultatGPUShared;
    u_char *d_ResultatGPUShared, *d_ResultatGPU, *d_Source;

	image<u_char> imgSource(height, width, &Source);
    image<u_char> imgResultat(height, width, &Resultat);
    image<u_char> imgResultatGPU(height, width, &ResultatGPU);
    image<u_char> imgResultatGPUShared(height, width, &ResultatGPUShared);
    
       
    auto fail = init_source_image(Source, image_filename, height, width);
    if (fail) {
        std::cout << "Chargement impossible de l'image" << std::endl;
        return 0;
    }
 
    cudaMalloc(&d_Source, height*width*sizeof(u_char));    
    cudaMalloc(&d_ResultatGPU, height*width*sizeof(u_char));    
    cudaMalloc(&d_ResultatGPUShared, height*width*sizeof(u_char));    

    cudaMemcpy(d_Source, Source[0], height*width*sizeof(u_char), cudaMemcpyHostToDevice);

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    for (auto it =0; it < ITER; it++) {
        cpu_sobel(Source, Resultat, height, width, nthreads);
    }
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration<double>(t1-t0).count()/ITER;

    dim3 threads(BLOCKDIM_X,BLOCKDIM_Y);
    dim3 blocks(width/BLOCKDIM_X,height/BLOCKDIM_Y);
    t0 = std::chrono::high_resolution_clock::now();
    for (auto it =0; it < ITER; it++) {
        gpu_sobel<<<blocks,threads>>>(d_Source, d_ResultatGPU, height, width);
        cudaDeviceSynchronize();
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration<double>(t1-t0).count()/ITER;
    cudaMemcpy(ResultatGPU[0], d_ResultatGPU, height*width*sizeof(u_char), cudaMemcpyDeviceToHost);

    dim3 blocks2(width/(BLOCKDIM_X-2),height/(BLOCKDIM_Y-2));
    // dim3 blocks2(2,2);
    t0 = std::chrono::high_resolution_clock::now();
    for (auto it =0; it < ITER; it++) {
        gpu_sobel_shared<<<blocks2,threads>>>(d_Source, d_ResultatGPUShared, height, width);
        cudaDeviceSynchronize();
    }
    t1 = std::chrono::high_resolution_clock::now();
    auto gpu_duration_shared = std::chrono::duration<double>(t1-t0).count()/ITER;

    cudaMemcpy(ResultatGPUShared[0], d_ResultatGPUShared, height*width*sizeof(u_char), cudaMemcpyDeviceToHost);

    std::cout << cpu_duration << " "  << gpu_duration << " "  << gpu_duration_shared << std::endl;

    #ifdef DEBUG
        image_filename=std::string("images/Resultats/Sobel_cpu.pgm");
        save_gray_level_image(&imgResultat, image_filename, height, width);
        image_filename=std::string("images/Resultats/Sobel_gpu.pgm");
        save_gray_level_image(&imgResultatGPU, image_filename, height, width);
        image_filename=std::string("images/Resultats/Sobel_gpu_shared.pgm");
        save_gray_level_image(&imgResultatGPUShared, image_filename, height, width);
    #endif
    
    return 0;
}

