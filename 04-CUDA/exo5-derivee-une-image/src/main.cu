#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"

#include "structural_elements.hpp"
#include "container.hpp"
#include "global_parameters.hpp"
#include "ios.hpp"
#include "omp.h"

#define DEBUG 1

// useful link https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/

// suppose width > height
int tomat(int i, int j, int width)
{
    return i*width + j;
}

int* tovec(int i, int width)
{
    int* res = (int*)malloc(2*sizeof(int));
    res[0] = i/width;
    res[1] = i%width;
    return res;
}

void transpose(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width, unsigned int nthreads)
{

    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            Resultat[j][i] = Source[i][j];
        }
    }
}

__global__ void transpose_gpu(u_char *Source, u_char *Resultat, unsigned int height, unsigned int width, unsigned int nthreads)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < height*width)
    {
        int x = i/width;
        int y = i%width;
        int k = width*y + x;
        Resultat[k] = Source[i];
    }
}

void transpose_parallel(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width, unsigned int nthreads)
{
#pragma omp parallel for
    for (auto i = 1; i < height; i++)
    {
        for (auto j = 1; j < width; j++)
        {
            Resultat[i][j] =  Source[j][i];
        }
    }
}


int main(int argc, char *argv[])
{
    unsigned int height, width;
    unsigned int nthreads = omp_get_max_threads();

    std::string image_filename(argv[1]);

    get_source_params(image_filename, &height, &width);
    std::cout << "size: width*height" << width << " " << height << std::endl;
    u_char **Source, **Resultat, **Resultat_gpu, *d_source, *d_resultat_gpu;

    image<u_char> imgSource(height, width, &Source);
    image<u_char> imgResultat(height, width, &Resultat);
    image<u_char> imgResultat_gpu(height, width, &Resultat_gpu);


    cudaMalloc(&d_source, width*height*sizeof(u_char)); 
    cudaMalloc(&d_resultat_gpu, width*height*sizeof(u_char));


    auto fail = init_source_image(Source, image_filename, height, width);
    if (fail)
    {
        std::cout << "Chargement impossible de l'image" << std::endl;
        return 0;
    }
    cudaMemcpy(d_source, Source, width*height*sizeof(u_char), cudaMemcpyHostToDevice);

    std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
    transpose(Source, Resultat, height, width, nthreads);
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "time seq: " << duration << std::endl;

    t0 = std::chrono::high_resolution_clock::now();
    // transpose_parallel(Source, Resultat, height, width, nthreads);
    t1 = std::chrono::high_resolution_clock::now();
    auto duration_p = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "time parallel: " << duration_p << std::endl;

    t0 = std::chrono::high_resolution_clock::now();
    int N = width*height;
    int k = 16;
    transpose_gpu<<<(N+k)/k, k>>>(d_source, d_resultat_gpu, height, width, nthreads);
    t1 = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "time parallel: " << duration_gpu << std::endl;

    std::cout << "time seq / parallel: " << duration/duration_p << std::endl;
    
#ifdef DEBUG
    image_filename = std::string("Transpose.pgm");
    save_gray_level_image(&imgResultat, image_filename,width,height);
#endif

    return 0;
}
