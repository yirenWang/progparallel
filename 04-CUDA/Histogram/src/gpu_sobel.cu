#include "gpu_sobel.cuh"
#include "parameters.cuh"
 
__global__ void gpu_sobel(u_char *Source, unsigned long long *Resultat, unsigned int height, unsigned int width) {

    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int globalIndex = i*width+j;
  //  printf("global %d \n", globalIndex);
  //  printf("trans %d \n", j*height+i);

    atomicAdd(&Resultat[Source[globalIndex]], 1);
}

__global__ void gpu_sobel_shared(u_char *Source, unsigned long long *Resultat, unsigned int height, unsigned int width) {
    
    // size of block fixed to be 16 * 16 = 256
    __shared__ unsigned long long resultat_local[256];
    
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = blockIdx.y*BLOCKDIM_Y + y;
    int j = blockIdx.x*BLOCKDIM_X + x;
    int globalIndex = i*width+j;
    int localIndex = x*BLOCKDIM_X + y;

    if (localIndex <= 255)
      resultat_local[localIndex] = 0;
    
    __syncthreads();
    
    if (i < height && j < width)
      atomicAdd(&resultat_local[Source[globalIndex]], 1);
    
    // wait for all threads in block to finish
    __syncthreads();

    // Place local resultat into global array 
    // 256 threads dans un block
    if (localIndex <= 255)
      atomicAdd(&Resultat[localIndex], resultat_local[localIndex]);
}