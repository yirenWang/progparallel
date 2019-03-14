#include "gpu_sobel.cuh"
#include "parameters.cuh"
 
__global__ void gpu_sobel(u_char *Source, u_char *Resultat, unsigned int height, unsigned int width) {

    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int globalIndex = i*width+j;
  //  printf("global %d \n", globalIndex);
//    printf("trans %d \n", j*height+i);

    Resultat[j*height+i]  = Source[globalIndex];
}

__global__ void gpu_sobel_shared(u_char *Source, u_char *Resultat, unsigned int height, unsigned int width) {
    __shared__ u_char tuile[BLOCKDIM_X][BLOCKDIM_Y+1];
    
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = blockIdx.y*BLOCKDIM_Y + y;
    int j = blockIdx.x*BLOCKDIM_X + x;
    
    //mainstream    
    tuile[x][y] = Source[i*width+j];
    __syncthreads();

    Resultat[j*height+i]  = tuile[x][y];
}