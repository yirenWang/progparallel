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

__global__ void gpu_sobel_shared2(u_char *Source, u_char *Resultat, unsigned int height, unsigned int width) {
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



__global__ void gpu_sobel_shared(u_char *Source, u_char *Resultat, unsigned int height, unsigned int width) {
  // Example : block dim = 32
  // Block row set to 8 by default 
  int BLOCK_ROWS = 8;
  __shared__ u_char tuile[BLOCKDIM_X][BLOCKDIM_Y]; // image de sortie

  int x = blockIdx.x * BLOCKDIM_X + threadIdx.x;
  int y = blockIdx.y * BLOCKDIM_X + threadIdx.y;

  for (int j = 0; j < BLOCKDIM_X; j += BLOCK_ROWS)
    tuile[threadIdx.y+j][threadIdx.x] = Source[(y+j)*width + x];
  
  __syncthreads();

  x = blockIdx.y * BLOCKDIM_X + threadIdx.x;  // transpose block offset
  y = blockIdx.x * BLOCKDIM_X + threadIdx.y;

  for (int j = 0; j < BLOCKDIM_X; j += BLOCK_ROWS)
  {
    Resultat[(y+j)*width + x] = tuile[threadIdx.x][threadIdx.y + j];
  }

}